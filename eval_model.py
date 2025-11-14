import sqlite3
import pandas as pd
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
import evaluate
from scipy.spatial.distance import cosine

DB_PATH = 'empathetic_conversations.db'
TABLE_NAME = 'conversations'
MODEL_DIR = 'models/flan-t5-empathetic'
MAX_INPUT_TOKENS = 128
MAX_TARGET_TOKENS = 64

INSTRUCTION_PREFIX = (
    'You are a kind, nonjudgmental friend. '
    'Reply with emotional support and one small practical suggestion. '
    'Do not give medical advice. The person says: '
)

print("Loading models...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR)
device = model.device
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
print(f"Models loaded on device: {device}")

print("Loading validation data...")
conn = sqlite3.connect(DB_PATH)
val_df = pd.read_sql(
    f'SELECT user_text, assistant_text FROM {TABLE_NAME} WHERE split=?',
    conn,
    params=('valid',)
)
conn.close()
print(f"Loaded {len(val_df)} validation examples")

rouge = evaluate.load('rouge')
model.eval()
n_eval = min(200, len(val_df))
inputs = val_df['user_text'].tolist()[:n_eval]
refs = val_df['assistant_text'].tolist()[:n_eval]

print(f"\nEvaluating on {n_eval} examples...")
preds = []
for i, text in enumerate(inputs):
    if (i + 1) % 50 == 0:
        print(f'  Progress: {i+1}/{n_eval}')
    
    prompt = INSTRUCTION_PREFIX + text
    enc = tokenizer(
        prompt,
        return_tensors='pt',
        truncation=True,
        max_length=MAX_INPUT_TOKENS
    )
    # Move tensors to same device as model
    enc = {k: v.to(device) for k, v in enc.items()}
    
    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=MAX_TARGET_TOKENS,
            do_sample=False
        )
    pred = tokenizer.decode(out[0], skip_special_tokens=True)
    preds.append(pred)

print("\nCalculating ROUGE...")
rouge_scores = rouge.compute(predictions=preds, references=refs)

print("Calculating semantic similarity...")
ref_embeddings = embedding_model.encode(refs, show_progress_bar=False)
pred_embeddings = embedding_model.encode(preds, show_progress_bar=False)

cosine_similarities = []
for ref_emb, pred_emb in zip(ref_embeddings, pred_embeddings):
    similarity = 1 - cosine(ref_emb, pred_emb)
    cosine_similarities.append(similarity)

avg_semantic_similarity = np.mean(cosine_similarities)

print(f"\n{'='*50}")
print(f"ROUGE-L: {rouge_scores['rougeL']:.4f}")
print(f"Semantic Similarity: {avg_semantic_similarity:.4f}")
print(f"{'='*50}")

# Show a few examples
print("\nSample predictions:")
for i in range(min(3, len(inputs))):
    print(f"\n--- Example {i+1} (Similarity: {cosine_similarities[i]:.3f}) ---")
    print(f"User: {inputs[i]}")
    print(f"Expected: {refs[i]}")
    print(f"Model: {preds[i]}")

