# rag_pipeline.py
import os
import re
import json
import torch
import nltk
import faiss
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# 1. Setup
nltk.download("punkt")
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.1"
embedder = SentenceTransformer("all-MiniLM-L6-v2")
unified_documents = []
unified_embeddings = []
unified_index = None
RELEVANCE_THRESHOLD = 0.325


# 2. Model loading
def load_model_and_tokenizer():
    print("🚀 Loading model...")
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch_dtype,
    )
    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            device_map="auto",
            torch_dtype=torch_dtype,
            quantization_config=quant_config,
            trust_remote_code=True,
        )
    except:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            device_map="cpu",
            torch_dtype=torch.float32,
            trust_remote_code=True,
        )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    return model, tokenizer


model, tokenizer = load_model_and_tokenizer()


# 3. Load and parse raw text file into DataFrame
def load_database(path="dataset/Crossbar_Database.txt"):
    with open(path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    qas = re.findall(r"Q:\s*(.*?)\nA:\s*(.*?)(?=\nQ:|\Z)", raw_text, re.DOTALL)
    data = [{"question": q.strip(), "answer": a.strip()} for q, a in qas]
    return pd.DataFrame(data)


# 4. Chunk and embed database
def chunk_and_embed(df):
    chunks, meta = [], []
    for _, row in df.iterrows():
        q, a = str(row.get("question", "")).strip(), str(row.get("answer", "")).strip()
        if not q or not a:
            continue
        text = f"Q: {q}\nA: {a}"
        sentences = sent_tokenize(text)
        buffer = []
        for sent in sentences:
            buffer.append(sent)
            if len(" ".join(buffer).split()) > 500:
                chunks.append(" ".join(buffer))
                meta.append({"question": q[:100]})
                buffer = []
        if buffer:
            chunks.append(" ".join(buffer))
            meta.append({"question": q[:100]})
    embs = embedder.encode(chunks, convert_to_numpy=True)
    embs = normalize(embs, axis=1)
    for t, m, e in zip(chunks, meta, embs):
        unified_documents.append({"text": t, "metadata": m})
        unified_embeddings.append(e)


# 5. Build FAISS index
def build_index():
    global unified_index
    dim = unified_embeddings[0].shape[0]
    index = faiss.IndexFlatIP(dim)
    index.add(np.vstack(unified_embeddings))
    unified_index = index


# 6. Handle user query
def ask_question(question, top_k=3):
    q_embed = embedder.encode([question], convert_to_numpy=True)
    q_embed = normalize(q_embed, axis=1)
    D, I = unified_index.search(q_embed, top_k * 2)
    relevant = []
    seen = set()
    for idx in I[0]:
        doc = unified_documents[idx]
        text = doc["text"]
        if text not in seen:
            seen.add(text)
            relevant.append(text)
        if len(relevant) == top_k:
            break
    context = "\n".join(relevant)
    prompt = (
        f"You are a helpful assistant. Use only the context below to answer.\n"
        f"Context:\n{context}\n\n"
        f"User: {question}\nAssistant:"
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=300,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id,
    )
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer.split("Assistant:")[-1].strip()


# 7. Save results
def save_response(user_input, answer):
    result = {
        "question": user_input,
        "answer": answer,
        "explanation": "Answer provided using contextual retrieval.",
    }
    with open("answer.txt", "w", encoding="utf-8") as f:
        f.write(answer)
    with open("result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print("✅ Saved answer.txt and result.json")


# === ENTRY POINT ===
if __name__ == "__main__":
    print("📥 Loading raw database text...")
    df = load_database()
    print(f"✅ Loaded {len(df)} Q&A pairs")
    print("🔗 Chunking & embedding...")
    chunk_and_embed(df)
    print("🔍 Building FAISS index...")
    build_index()
    question = os.getenv("QUESTION", "What is Crossbar?")
    print(f"💬 User: {question}")
    answer = ask_question(question)
    print(f"🤖 Answer: {answer}")
    save_response(question, answer)
