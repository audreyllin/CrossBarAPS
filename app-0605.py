from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import json
import uuid
from datetime import datetime
import logging
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Configuration
DOCUMENT_DIR = "documents"
METADATA_FILE = os.path.join(DOCUMENT_DIR, "metadata.json")
CONTEXT_FILE = os.path.join(DOCUMENT_DIR, "context.json")

def init_storage():
    """Initialize document storage directories"""
    os.makedirs(DOCUMENT_DIR, exist_ok=True)
    
    # Create metadata file if not exists
    if not os.path.exists(METADATA_FILE):
        with open(METADATA_FILE, 'w') as f:
            json.dump({"documents": []}, f)
    
    # Create context file if not exists
    if not os.path.exists(CONTEXT_FILE):
        with open(CONTEXT_FILE, 'w') as f:
            json.dump({
                "currentFocus": {
                    "document": None,
                    "topic": "General",
                    "entities": [],
                    "keywords": []
                }
            }, f)

def load_metadata():
    """Load document metadata from file"""
    try:
        with open(METADATA_FILE, 'r') as f:
            return json.load(f)
    except:
        return {"documents": []}

def save_metadata(metadata):
    """Save document metadata to file"""
    with open(METADATA_FILE, 'w') as f:
        json.dump(metadata, f, indent=2)

def load_context():
    """Load context state from file"""
    try:
        with open(CONTEXT_FILE, 'r') as f:
            return json.load(f)
    except:
        return {
            "currentFocus": {
                "document": None,
                "topic": "General",
                "entities": [],
                "keywords": []
            }
        }

def save_context(context):
    """Save context state to file"""
    with open(CONTEXT_FILE, 'w') as f:
        json.dump(context, f, indent=2)

def extract_keywords(content):
    """Basic keyword extraction"""
    from collections import defaultdict
    import re
    
    # Simple word frequency analysis
    words = re.findall(r'\b\w{4,}\b', content.lower())
    word_counts = defaultdict(int)
    for word in words:
        word_counts[word] += 1
    
    # Return top 10 keywords
    sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
    return {word: {"count": count} for word, count in sorted_words[:10]}

def generate_summary(content, max_length=200):
    """Generate document summary"""
    sentences = content.split('.')
    summary = ""
    for sentence in sentences:
        if len(summary) + len(sentence) < max_length:
            summary += sentence + '.'
        else:
            break
    return summary.strip()

@app.route('/')
def serve_index():
    return send_from_directory('.', 'rag-0605.html')

@app.route('/<path:filename>')
def serve_file(filename):
    return send_from_directory('.', filename)

@app.route("/initial-context", methods=["GET"])
def get_initial_context():
    """Provide initial context to frontend"""
    try:
        context = load_context()
        metadata = load_metadata()
        
        # If no documents, add sample documents
        if not metadata["documents"]:
            sample_docs = [
                {
                    "id": "doc-001",
                    "name": "arXiv:2307.12874",
                    "type": "research_paper",
                    "source": "https://arxiv.org/pdf/2307.12874",
                    "content": "This paper presents a comprehensive evaluation of post-quantum cryptographic algorithms...",
                    "keywords": extract_keywords("This paper presents a comprehensive evaluation of post-quantum cryptographic algorithms..."),
                    "summary": "Evaluation of post-quantum cryptographic algorithms for hardware wallets",
                    "created_at": datetime.now().isoformat()
                },
                {
                    "id": "doc-002",
                    "name": "CROSSBAR Security Whitepaper",
                    "type": "whitepaper",
                    "source": "internal",
                    "content": "The CROSSBAR Security Whitepaper outlines our approach to hardware wallet security...",
                    "keywords": extract_keywords("The CROSSBAR Security Whitepaper outlines our approach to hardware wallet security..."),
                    "summary": "Hardware wallet security with quantum-resistant cryptography",
                    "created_at": datetime.now().isoformat()
                }
            ]
            metadata["documents"] = sample_docs
            save_metadata(metadata)
            
            # Update context with first document
            context["currentFocus"]["document"] = "doc-001"
            context["currentFocus"]["topic"] = "Quantum-Resistant Algorithms"
            save_context(context)
        
        return jsonify({
            "documents": metadata["documents"],
            "currentFocus": context["currentFocus"]
        })
    except Exception as e:
        logger.error(f"Error in /initial-context: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/process-document", methods=["POST"])
def process_document():
    """Process document content and extract metadata"""
    try:
        data = request.json
        content = data.get("content", "")
        
        # Process document content
        keywords = extract_keywords(content)
        summary = generate_summary(content)
        
        return jsonify({
            "keywords": keywords,
            "summary": summary,
            "entities": list(keywords.keys())[:5],  # Top 5 entities
            "processed_at": datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Document processing error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/retrieve-context", methods=["POST"])
def retrieve_context():
    """Retrieve relevant context based on query"""
    try:
        data = request.json
        query = data.get("query", "")
        
        # In a real implementation, this would use semantic search
        # For demo, we return mock results
        return jsonify({
            "documentId": "doc-001",
            "topic": "Quantum-Resistant Algorithms",
            "entities": ["quantum-resistant", "hardware wallets", "cryptography"],
            "keywords": ["security", "implementation", "analysis"],
            "suggestedFocus": {
                "document": "doc-001",
                "topic": "Quantum-Resistant Algorithms"
            }
        })
    except Exception as e:
        logger.error(f"Context retrieval error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/update-context", methods=["POST"])
def update_context():
    """Update context state from frontend"""
    try:
        data = request.json
        new_context = data.get("context", {})
        
        # Validate and update context
        if "currentFocus" in new_context:
            current_context = load_context()
            current_context["currentFocus"] = {
                **current_context["currentFocus"],
                **new_context["currentFocus"]
            }
            save_context(current_context)
            return jsonify({"success": True})
        return jsonify({"error": "Invalid context format"}), 400
    except Exception as e:
        logger.error(f"Context update error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/add-document", methods=["POST"])
def add_document():
    """Add a new document to the system"""
    try:
        data = request.json
        document = data.get("document", {})
        
        # Validate document
        if not all(key in document for key in ["name", "type", "content"]):
            return jsonify({"error": "Missing required document fields"}), 400
        
        # Generate document ID
        doc_id = f"doc-{uuid.uuid4().hex[:8]}"
        document["id"] = doc_id
        document["created_at"] = datetime.now().isoformat()
        
        # Process document if needed
        if "keywords" not in document:
            document["keywords"] = extract_keywords(document["content"])
        if "summary" not in document:
            document["summary"] = generate_summary(document["content"])
        
        # Add to metadata
        metadata = load_metadata()
        metadata["documents"].append(document)
        save_metadata(metadata)
        
        return jsonify({
            "success": True,
            "documentId": doc_id
        })
    except Exception as e:
        logger.error(f"Document addition error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/status', methods=['GET', 'OPTIONS'])
def status_check():
    """Health check endpoint"""
    if request.method == 'OPTIONS':
        # Handle preflight request
        response = jsonify({'status': 'ok'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', '*')
        response.headers.add('Access-Control-Allow-Methods', '*')
        return response
    
    return jsonify({
        "status": "ok",
        "message": "Service is running",
        "timestamp": datetime.now().isoformat(),
        "system": "Document Assistant API",
        "version": "1.0.0"
    }), 200

@app.route('/ask', methods=['POST'])
def ask_question():
    """Handle user questions with document context"""
    try:
        data = request.json
        query = data.get("query", "")
        context = data.get("context", {})
        
        # In production: Use NLP to generate response
        return jsonify({
            "answer": f"Received your query: '{query}' about document {context.get('documentId', 'N/A')}",
            "suggestions": [
                "Can you elaborate on this?",
                "What specific aspect are you interested in?",
                "Would you like to compare this to other documents?"
            ]
        })
    except Exception as e:
        logger.error(f"Question handling error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file uploads"""
    try:
        # In production: Process uploaded file
        return jsonify({
            "success": True,
            "summary": "Document processed successfully",
            "keywords": {"security": {"count": 15}, "cryptography": {"count": 12}}
        })
    except Exception as e:
        logger.error(f"File upload error: {e}")
        return jsonify({"error": str(e)}), 500

@app.after_request
def add_cors_headers(response):
    if request.method == 'OPTIONS':
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
        response.headers['Access-Control-Max-Age'] = '86400'
    return response

if __name__ == "__main__":
    init_storage()
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True,
        threaded=True
    )
