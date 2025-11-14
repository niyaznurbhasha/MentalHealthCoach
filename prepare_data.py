import sqlite3
import re
import pandas as pd
from datasets import load_dataset

DB_PATH = "empathetic_conversations.db"
TABLE_NAME = "conversations"
MIN_LEN = 5
MAX_LEN = 80


def clean_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\n", " ").replace("\t", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()
def extract_pairs_from_conversation(conv):
    """
    conv: list of {role: 'user'|'assistant', content: str}
    Return list of (user_text, assistant_text) pairs where assistant replies to previous user.
    """
    pairs = []
    for i in range(1, len(conv)):
        curr = conv[i]
        prev = conv[i - 1]
        if curr["role"] == "assistant" and prev["role"] == "user":
            user_text = clean_text(prev["content"])
            assistant_text = clean_text(curr["content"])
            if user_text and assistant_text:
                pairs.append((user_text, assistant_text))
    return pairs

def main():
    ds = load_dataset("Estwld/empathetic_dialogues_llm")

    all_rows = []
    for split in ["train", "valid", "test"]:
        for ex in ds[split]:
            conv = ex["conversations"]
            emotion = ex.get("emotion", "")
            situation = ex.get("situation", "")
            pairs = extract_pairs_from_conversation(conv)
            for user_text, assistant_text in pairs:
                all_rows.append(
                    {
                        "user_text": user_text,
                        "assistant_text": assistant_text,
                        "emotion": emotion,
                        "situation": situation,
                        "split": split,
                    }
                )

    df = pd.DataFrame(all_rows)

    df["tokens_in"] = df["user_text"].str.split().str.len()
    df["tokens_out"] = df["assistant_text"].str.split().str.len()
    df = df[(df["tokens_in"] >= MIN_LEN) & (df["tokens_in"] <= MAX_LEN)]

    conn = sqlite3.connect(DB_PATH)
    df.to_sql(TABLE_NAME, conn, if_exists="replace", index=False)
    conn.close()

    print(f"Saved {len(df)} rows to {DB_PATH}:{TABLE_NAME}")

if __name__ == "__main__":
    main()
