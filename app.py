from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.llms import HuggingFaceHub
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app, resources={
    r"/*": {"origins": "*"}  # Allow all origins for development
})

# Configuration
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DOCUMENT_DIR = "documents"
TEMP_DOWNLOAD_DIR = "temp_downloads"

# Set USER_AGENT for web requests
os.environ['USER_AGENT'] = 'MyLocalRAG/1.0'

# Initialize models
try:
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    llm = HuggingFaceEndpoint(
        repo_id="google/flan-t5-large",
        huggingfacehub_api_token="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"
    )
except Exception as e:
    print(f"Error initializing models: {e}")
    raise

# Document store setup
document_store = None
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

def initialize_document_store():
    global document_store
    if os.path.exists(f"{DOCUMENT_DIR}/faiss_index"):
        document_store = FAISS.load_local(DOCUMENT_DIR, embeddings)
    else:
        document_store = FAISS.from_texts(["System initialized"], embeddings)
        document_store.save_local(DOCUMENT_DIR)
    
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=document_store.as_retriever(),
        return_source_documents=True
    )

qa_chain = initialize_document_store()

def process_web_content(url):
    """Download and process web content without storing permanently"""
    try:
        os.makedirs(TEMP_DOWNLOAD_DIR, exist_ok=True)
        loader = WebBaseLoader(url)
        documents = loader.load()
        splits = text_splitter.split_documents(documents)
        
        temp_store = FAISS.from_documents(splits, embeddings)
        
        return {
            "temp_store": temp_store,
            "documents": splits
        }
    except Exception as e:
        print(f"Error processing web content: {e}")
        raise

@app.route('/')
def serve_index():
    return send_from_directory('.', 'rag-0605.html')

@app.route('/<path:filename>')
def serve_file(filename):
    return send_from_directory('.', filename)

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
            "documents": [doc.metadata.get("source", "local") for doc in result["source_documents"]],
            "source": "local"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/upload", methods=["POST"])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Empty filename"}), 400
    
    try:
        temp_path = os.path.join(TEMP_DOWNLOAD_DIR, file.filename)
        file.save(temp_path)
        
        with open(temp_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        splits = text_splitter.split_text(content)
        
        global document_store
        document_store.add_texts(
            splits,
            metadatas=[{"source": file.filename} for _ in splits]
        )
        document_store.save_local(DOCUMENT_DIR)
        os.remove(temp_path)
        
        return jsonify({
            "success": True,
            "filename": file.filename,
            "chunks": len(splits)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/add_url", methods=["POST"])
def add_url():
    data = request.get_json()
    url = data.get("url")
    if not url:
        return jsonify({"error": "No URL provided"}), 400
    
    try:
        web_content = process_web_content(url)
        web_qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=web_content["temp_store"].as_retriever()
        )
        result = web_qa({"query": "Summarize the key points"})
        
        return jsonify({
            "answer": result["result"],
            "source": url,
            "temporary": True
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# NEW ENDPOINTS FOR TESTING AND DEBUGGING
@app.route("/test_retrieval", methods=["POST"])
def test_retrieval():
    """Manually test retrieval with a specific query"""
    try:
        data = request.get_json()
        query = data.get("query", "supersingular isogenies")  # Default to paper topic
        k = data.get("k", 3)  # Number of results to return
        
        # Direct retrieval without LLM
        docs = document_store.similarity_search(query, k=k)
        
        return jsonify({
            "query": query,
            "results": [{
                "content": doc.page_content,
                "source": doc.metadata.get("source", "unknown"),
                "score": None  # FAISS doesn't return scores by default
            } for doc in docs]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/reingest_test", methods=["POST"])
def reingest_test_document():
    """Re-ingest a small test document to verify the pipeline"""
    try:
        test_text = """
        Quantum-resistant cryptography uses mathematical constructions like 
        supersingular isogenies to create encryption that cannot be broken by 
        quantum computers. The paper arXiv:2307.12874 proposes a new 
        public-key cryptosystem based on these principles.
        """
        
        splits = text_splitter.split_text(test_text)
        
        global document_store
        document_store.add_texts(
            splits,
            metadatas=[{"source": "test_document"} for _ in splits]
        )
        document_store.save_local(DOCUMENT_DIR)
        
        return jsonify({
            "success": True,
            "message": "Test document re-ingested",
            "chunks": len(splits)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/get_document_stats", methods=["GET"])
def get_document_stats():
    """Get statistics about the stored documents"""
    try:
        if not os.path.exists(f"{DOCUMENT_DIR}/faiss_index"):
            return jsonify({"error": "No document store exists yet"}), 404
            
        index = FAISS.load_local(DOCUMENT_DIR, embeddings)
        return jsonify({
            "document_count": index.index.ntotal,
            "embedding_dim": index.index.d,
            "index_type": str(type(index.index))
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    os.makedirs(DOCUMENT_DIR, exist_ok=True)
    os.makedirs(TEMP_DOWNLOAD_DIR, exist_ok=True)
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True,
        ssl_context='adhoc'
    )