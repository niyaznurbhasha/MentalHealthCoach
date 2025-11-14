import sqlite3
import random
import numpy as np
import pandas as pd
import torch

from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
import evaluate

DB_PATH = "empathetic_conversations.db"
TABLE_NAME = "conversations"
MODEL_NAME = "google/flan-t5-small"
OUTPUT_DIR = "models/flan-t5-empathetic"

MAX_INPUT_TOKENS = 128
MAX_TARGET_TOKENS = 64

INSTRUCTION_PREFIX = (
    "You are a kind, nonjudgmental friend. "
    "Reply with emotional support and one small practical suggestion. "
    "Do not give medical advice. The person says: "
)

rouge = evaluate.load("rouge")


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_split(split: str) -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql(
        f"SELECT user_text, assistant_text FROM {TABLE_NAME} WHERE split=?",
        conn,
        params=(split,),
    )
    conn.close()
    return df


def main():
    set_seed(42)

    # 1) load dataframes
    train_df = load_split("train")
    val_df = load_split("valid")  # split name from prepare_data.py

    # 2) convert to HF datasets
    train_ds = Dataset.from_pandas(train_df)
    val_ds = Dataset.from_pandas(val_df)
    ds = DatasetDict({"train": train_ds, "validation": val_ds})

    # 3) tokenizer + model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    def preprocess(batch):
        inputs = [INSTRUCTION_PREFIX + x for x in batch["user_text"]]
        targets = batch["assistant_text"]
        model_inputs = tokenizer(
            inputs,
            max_length=MAX_INPUT_TOKENS,
            truncation=True,
        )
        # Use text_target instead of deprecated as_target_tokenizer()
        labels = tokenizer(
            text_target=targets,
            max_length=MAX_TARGET_TOKENS,
            truncation=True,
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized = ds.map(
        preprocess,
        batched=True,
        remove_columns=["user_text", "assistant_text"],
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    # 4) training args: use eval_loss as metric for early stopping / best model
    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate=5e-5,
        weight_decay=0.01,
        eval_strategy="epoch",  # Fixed: was evaluation_strategy in older versions
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=50,
        label_smoothing_factor=0.1,
        fp16=False,
        gradient_accumulation_steps=1,
        max_grad_norm=1.0,   # gradient clipping
        warmup_ratio=0.1,    # LR warmup
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    # 5) train (uses eval_loss for early stopping / best model)
    trainer.train()

    # best model is already loaded if load_best_model_at_end=True
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Saved model to {OUTPUT_DIR}")

    # 6) manual ROUGE eval on a small validation subset
    model.eval()
    device = model.device  # Get the device the model is on
    n_eval = min(200, len(val_df))
    inputs = val_df["user_text"].tolist()[:n_eval]
    refs = val_df["assistant_text"].tolist()[:n_eval]

    preds = []
    for text in inputs:
        prompt = INSTRUCTION_PREFIX + text
        enc = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_INPUT_TOKENS,
        )
        # Move input tensors to the same device as the model
        enc = {k: v.to(device) for k, v in enc.items()}
        
        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=MAX_TARGET_TOKENS,
                do_sample=False,
            )
        pred = tokenizer.decode(out[0], skip_special_tokens=True)
        preds.append(pred)

    scores = rouge.compute(predictions=preds, references=refs)
    print("Manual ROUGE-L on validation subset:", scores["rougeL"])


if __name__ == "__main__":
    main()
