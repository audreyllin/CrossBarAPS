from flask import Flask, request, jsonify
from flask_cors import CORS
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
import os

app = Flask(__name__)
CORS(app)

# Configuration
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
OPENAI_MODEL = "gpt-4"
DOCUMENT_DIR = "documents"

# Initialize models
try:
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    embeddings = OpenAIEmbeddings()
    llm = OpenAI(model_name=OPENAI_MODEL, temperature=0)
except Exception as e:
    print(f"Error initializing models: {e}")
    raise

# Document store setup
document_store = None
documents = {}

def initialize_document_store():
    global document_store
    if os.path.exists(f"{DOCUMENT_DIR}/faiss_index"):
        document_store = FAISS.load_local(DOCUMENT_DIR, embeddings)
    else:
        document_store = FAISS.from_texts([""], embeddings)
        document_store.save_local(DOCUMENT_DIR)
    
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=document_store.as_retriever(),
        return_source_documents=True
    )

qa_chain = initialize_document_store()

@app.route("/ask", methods=["POST"])
def ask():
    try:
        data = request.get_json()
        query = data.get("query")
        if not query:
            return jsonify({"error": "No query provided"}), 400
        
        result = qa_chain({"query": query})
        return jsonify({
            "answer": result["result"],
            "documents": [doc.metadata.get("source", "") for doc in result["source_documents"]]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    os.makedirs(DOCUMENT_DIR, exist_ok=True)
    app.run(host="0.0.0.0", port=5000, debug=True)