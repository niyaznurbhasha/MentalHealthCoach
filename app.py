import sqlite3
import pandas as pd
import numpy as np
import torch
import streamlit as st
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
import evaluate

DB_PATH = "empathetic_conversations.db"
TABLE_NAME = "conversations"
MODEL_PATH = "models/flan-t5-empathetic"

INSTRUCTION_PREFIX = (
    "You are a kind, nonjudgmental friend. "
    "Reply with emotional support and one small practical suggestion. "
    "Do not give medical advice. The person says: "
)

def get_db():
    """Create a new database connection (don't cache - SQLite is not thread-safe)"""
    return sqlite3.connect(DB_PATH, check_same_thread=False)

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)
    return tokenizer, model


def generate(text, conversation_history=""):
    tokenizer, model = load_model()
    # Include conversation history for context
    if conversation_history:
        prompt = INSTRUCTION_PREFIX + conversation_history + "\nUser: " + text
    else:
        prompt = INSTRUCTION_PREFIX + text
    
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(
        **inputs,
        max_new_tokens=150,
        do_sample=True,
        top_p=0.9,
        temperature=0.7,
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def main():
    st.title("Empathetic Buddy - Data Explorer and Demo")

    tab1, tab2, tab3 = st.tabs(["Dataset Explorer", "Model Results", "Chat with Model"])

    with tab1:
        st.header("SQLite dataset overview")
        
        conn = get_db()
        counts = pd.read_sql(
            f"SELECT split, COUNT(*) AS n FROM {TABLE_NAME} GROUP BY split",
            conn,
        )
        conn.close()
        st.write("Rows per split:")
        st.dataframe(counts)

        st.subheader("Token Length Distributions")
        st.write("These distributions informed the choice of MAX_INPUT_TOKENS=128 and MAX_TARGET_TOKENS=64")
        
        conn = get_db()
        tokens = pd.read_sql(
            f"SELECT tokens_in, tokens_out FROM {TABLE_NAME} WHERE split='train'",
            conn,
        )
        conn.close()
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.hist(tokens["tokens_in"], bins=30, color='skyblue', edgecolor='black')
            ax.axvline(128, color='red', linestyle='--', label='MAX_INPUT=128')
            ax.set_xlabel("Input token count")
            ax.set_ylabel("Frequency")
            ax.set_title("Input Length Distribution")
            ax.legend()
            st.pyplot(fig)
        
        with col2:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.hist(tokens["tokens_out"], bins=30, color='lightcoral', edgecolor='black')
            ax.axvline(64, color='red', linestyle='--', label='MAX_TARGET=64')
            ax.set_xlabel("Output token count")
            ax.set_ylabel("Frequency")
            ax.set_title("Output Length Distribution")
            ax.legend()
            st.pyplot(fig)

        st.subheader("Sample training pairs")
        min_len = st.slider("Min tokens", 0, 200, 10)
        max_len = st.slider("Max tokens", 10, 200, 60)
        limit = st.slider("Number of rows", 1, 20, 5)

        query = (
            f"SELECT user_text, assistant_text, tokens_in "
            f"FROM {TABLE_NAME} "
            f"WHERE split='train' AND tokens_in BETWEEN ? AND ? "
            f"LIMIT ?"
        )
        conn = get_db()
        sample = pd.read_sql(query, conn, params=(min_len, max_len, limit))
        conn.close()
        st.dataframe(sample)

    with tab2:
        st.header("Model Evaluation Results")
        st.write("Evaluate model on validation data with ROUGE-L and semantic similarity metrics.")
        
        n_samples = st.number_input("Number of validation samples", 10, 200, 50)
        
        if st.button("Run Evaluation", type="primary"):
            with st.spinner("Evaluating..."):
                # Load validation data
                conn = get_db()
                val_df = pd.read_sql(
                    f"SELECT user_text, assistant_text FROM {TABLE_NAME} WHERE split='valid' LIMIT ?",
                    conn, params=(n_samples,)
                )
                conn.close()
                
                # Generate predictions
                tokenizer, model = load_model()
                device = model.device
                preds, refs = [], []
                
                for _, row in val_df.iterrows():
                    prompt = INSTRUCTION_PREFIX + row['user_text']
                    enc = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=128)
                    enc = {k: v.to(device) for k, v in enc.items()}
                    with torch.no_grad():
                        out = model.generate(**enc, max_new_tokens=64, do_sample=False)
                    preds.append(tokenizer.decode(out[0], skip_special_tokens=True))
                    refs.append(row['assistant_text'])
                
                # Calculate metrics
                rouge = evaluate.load('rouge')
                rouge_score = rouge.compute(predictions=preds, references=refs)['rougeL']
                
                embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                ref_emb = embedding_model.encode(refs, show_progress_bar=False)
                pred_emb = embedding_model.encode(preds, show_progress_bar=False)
                sims = [1 - cosine(r, p) for r, p in zip(ref_emb, pred_emb)]
                
                # Display results
                col1, col2 = st.columns(2)
                col1.metric("ROUGE-L", f"{rouge_score:.4f}")
                col2.metric("Avg Semantic Similarity", f"{np.mean(sims):.4f}")
                
                # Similarity distribution
                fig, ax = plt.subplots(figsize=(8, 3))
                ax.hist(sims, bins=20, color='skyblue', edgecolor='black')
                ax.axvline(np.mean(sims), color='red', linestyle='--', label=f'Mean: {np.mean(sims):.3f}')
                ax.set_xlabel('Semantic Similarity'); ax.set_ylabel('Count')
                ax.set_title('Semantic Similarity Distribution'); ax.legend()
                st.pyplot(fig)
                
                # Show sample predictions
                st.subheader("Sample Predictions")
                for i in range(min(3, len(preds))):
                    with st.expander(f"Example {i+1} - Similarity: {sims[i]:.3f}"):
                        st.write(f"**Input:** {val_df.iloc[i]['user_text']}")
                        st.write(f"**Expected:** {refs[i]}")
                        st.write(f"**Model:** {preds[i]}")

    with tab3:
        st.header("Talk to the fine tuned model")
        
        # Initialize chat history in session state
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("What's on your mind?"):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Build conversation history (last 3 exchanges for context)
            history = ""
            for msg in st.session_state.messages[-6:]:  # Last 3 user + 3 assistant
                if msg["role"] == "user":
                    history += f"User: {msg['content']}\n"
                else:
                    history += f"Friend: {msg['content']}\n"
            
            # Generate response
            with st.chat_message("assistant"):
                reply = generate(prompt, history)
                st.markdown(reply)
                st.session_state.messages.append({"role": "assistant", "content": reply})
        
        # Clear chat button
        if st.button("Clear Chat"):
            st.session_state.messages = []
            st.rerun()

if __name__ == "__main__":
    main()
