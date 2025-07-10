from flask import Flask, request, jsonify, render_template
import os
import logging
import uuid
import tiktoken
import json
import PyPDF2
import pdfplumber
from docx import Document
import hashlib
import re
import sqlite3
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
from werkzeug.utils import secure_filename

# Initialize app and paths
current_dir = os.path.dirname(os.path.abspath(__file__))
template_path = os.path.join(current_dir, "templates")
DB_FILE = os.path.join(current_dir, "memory.db")
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
ALLOWED_EXTENSIONS = {
    "txt",
    "pdf",
    "doc",
    "docx",
    "ppt",
    "pptx",
    "jpg",
    "jpeg",
    "png",
    "mp4",
}

app = Flask(__name__, template_folder=template_path)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
logging.basicConfig(level=logging.INFO)


# Initialize database
def init_db():
    with sqlite3.connect(DB_FILE) as db:
        db.execute(
            """
            CREATE TABLE IF NOT EXISTS context_vectors (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                context_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                vector TEXT NOT NULL,
                model TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """
        )
        db.execute(
            """
            CREATE TABLE IF NOT EXISTS session_contexts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                context_id TEXT NOT NULL,
                content TEXT NOT NULL,
                source_type TEXT,
                filename TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """
        )
        db.execute(
            """
            CREATE TABLE IF NOT EXISTS conversation_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """
        )


init_db()

# Token counting
encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")


def count_tokens(text):
    return len(encoding.encode(text))


# Text extraction from files
def extract_text_from_file(filepath, filename):
    ext = filename.split(".")[-1].lower()
    text = ""

    try:
        if ext == "pdf":
            with pdfplumber.open(filepath) as pdf:
                text = "\n".join(
                    [page.extract_text() for page in pdf.pages if page.extract_text()]
                )
        elif ext in ["doc", "docx"]:
            doc = Document(filepath)
            text = "\n".join([para.text for para in doc.paragraphs])
        elif ext in ["ppt", "pptx"]:
            # Placeholder for PPT extraction - would require additional libraries
            text = "PPT content extraction not implemented"
        elif ext in ["jpg", "jpeg", "png"]:
            # Placeholder for image OCR - would require additional libraries
            text = "Image OCR not implemented"
        elif ext == "mp4":
            # Placeholder for video transcription - would require additional libraries
            text = "Video transcription not implemented"
        else:  # txt and others
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read()
    except Exception as e:
        logging.error(f"Text extraction failed: {str(e)}")
        text = f"Error extracting text: {str(e)}"

    return text[:10000]  # Limit to 10k characters


# GPT Embedding with model selection
def generate_vector(text, api_key, model="text-embedding-3-large"):
    try:
        client = OpenAI(api_key=api_key)
        response = client.embeddings.create(
            model=model, input=[text[:8192]]  # Truncate to model limit
        )
        return np.array(response.data[0].embedding)
    except Exception as e:
        logging.error(f"Embedding generation failed: {str(e)}")
        # Return zero vector of appropriate dimension
        dim = 3072 if model == "text-embedding-3-large" else 1536
        return np.zeros(dim)


# GPT-based context retrieval
def get_context_vectors(session_id, model):
    with sqlite3.connect(DB_FILE) as db:
        rows = db.execute(
            "SELECT context_id, vector FROM context_vectors WHERE session_id = ? AND model = ?",
            (session_id, model),
        ).fetchall()
    return [(row[0], np.array(json.loads(row[1]))) for row in rows]


def get_relevant_contexts(
    question, session_contexts, session_id, api_key, model, top_n=3
):
    if not question or not session_contexts:
        return []

    stored_vectors = get_context_vectors(session_id, model)
    if not stored_vectors:
        return []

    question_vector = generate_vector(question, api_key, model=model)
    similarities = []
    for context_id, ctx_vec in stored_vectors:
        try:
            similarity = cosine_similarity([question_vector], [ctx_vec])[0][0]
            similarities.append((context_id, similarity))
        except Exception as e:
            logging.error(f"Similarity calculation error: {str(e)}")
            similarities.append((context_id, 0))

    similarities.sort(key=lambda x: x[1], reverse=True)
    top_contexts = [ctx_id for ctx_id, _ in similarities[:top_n]]
    return [ctx for ctx in session_contexts if ctx["id"] in top_contexts]


# Debug/test route to inspect vector generation
@app.route("/api/debug_vector", methods=["POST"])
def debug_vector():
    api_key = request.headers.get("Authorization", "").replace("Bearer ", "")
    if not api_key:
        return jsonify({"error": "Missing API key"}), 401

    data = request.get_json()
    text = data.get("text", "").strip()
    model = data.get("model", "text-embedding-3-large")

    if not text:
        return jsonify({"error": "Empty input text"}), 400

    try:
        vector = generate_vector(text, api_key, model=model)
        return jsonify(
            {
                "model": model,
                "embedding_dim": len(vector),
                "preview": vector[:10].tolist(),
                "vector_hash": hashlib.sha256(vector.tobytes()).hexdigest()[:16],
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Clear all vectors for a given session
@app.route("/api/clear_vectors", methods=["POST"])
def clear_vectors():
    data = request.get_json()
    session_id = data.get("sessionId", "")
    if not session_id:
        return jsonify({"error": "Missing session ID"}), 400

    try:
        with sqlite3.connect(DB_FILE) as db:
            db.execute(
                "DELETE FROM context_vectors WHERE session_id = ?", (session_id,)
            )
            db.commit()
        return jsonify(
            {
                "status": "success",
                "message": f"All vectors cleared for session {session_id}",
            }
        )
    except Exception as e:
        logging.error(f"Failed to clear vectors: {str(e)}")
        return jsonify({"error": str(e)}), 500


# Upload context handler
@app.route("/api/upload_context", methods=["POST"])
def upload_context():
    api_key = request.headers.get("Authorization", "").replace("Bearer ", "")
    if not api_key:
        return jsonify({"error": "Missing API key"}), 401

    session_id = request.form.get("sessionId")
    model = request.form.get("model", "text-embedding-3-large")

    if not session_id:
        return jsonify({"error": "Missing session ID"}), 400

    # Handle text context
    if "text" in request.form:
        text_content = request.form["text"]
        if not text_content:
            return jsonify({"error": "Empty text content"}), 400

        context_id = f"text_{hashlib.md5(text_content.encode()).hexdigest()[:10]}"

        # Generate and store vector
        vector = generate_vector(text_content, api_key, model=model)
        with sqlite3.connect(DB_FILE) as db:
            db.execute(
                "INSERT INTO context_vectors (context_id, session_id, vector, model) VALUES (?, ?, ?, ?)",
                (context_id, session_id, json.dumps(vector.tolist()), model),
            )
            db.execute(
                "INSERT INTO session_contexts (session_id, context_id, content, source_type) VALUES (?, ?, ?, ?)",
                (session_id, context_id, text_content, "text"),
            )
            db.commit()

        # Extract key concepts (simplified)
        words = re.findall(r"\b\w{4,}\b", text_content.lower())
        stop_words = set(
            ["this", "that", "which", "with", "from", "your", "have", "more", "about"]
        )
        top_words = [
            word
            for word, count in Counter(words).most_common(5)
            if word not in stop_words and len(word) > 3
        ]
        concepts = [word.capitalize() for word in top_words]

        return jsonify(
            {"status": "success", "context_id": context_id, "concepts": concepts}
        )

    # Handle file context
    elif "file" in request.files:
        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No selected file"}), 400

        if not allowed_file(file.filename):
            return jsonify({"error": "File type not allowed"}), 400

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        # Extract text from file
        text_content = extract_text_from_file(filepath, filename)
        context_id = f"file_{hashlib.md5(filename.encode()).hexdigest()[:10]}"

        # Generate and store vector
        vector = generate_vector(text_content, api_key, model=model)
        with sqlite3.connect(DB_FILE) as db:
            db.execute(
                "INSERT INTO context_vectors (context_id, session_id, vector, model) VALUES (?, ?, ?, ?)",
                (context_id, session_id, json.dumps(vector.tolist()), model),
            )
            db.execute(
                "INSERT INTO session_contexts (session_id, context_id, content, source_type, filename) VALUES (?, ?, ?, ?, ?)",
                (session_id, context_id, text_content, "file", filename),
            )
            db.commit()

        # Extract key concepts
        words = re.findall(r"\b\w{4,}\b", text_content.lower())
        stop_words = set(
            ["this", "that", "which", "with", "from", "your", "have", "more", "about"]
        )
        top_words = [
            word
            for word, count in Counter(words).most_common(5)
            if word not in stop_words and len(word) > 3
        ]
        concepts = [word.capitalize() for word in top_words]

        return jsonify(
            {"status": "success", "context_id": context_id, "concepts": concepts}
        )

    return jsonify({"error": "No valid context provided"}), 400


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# Question answering endpoint
@app.route("/api/ask", methods=["POST"])
def ask_question():
    api_key = request.headers.get("Authorization", "").replace("Bearer ", "")
    if not api_key:
        return jsonify({"error": "Missing API key"}), 401

    data = request.get_json()
    question = data.get("question", "").strip()
    session_id = data.get("sessionId", "")
    model = data.get("model", "text-embedding-3-large")

    if not question:
        return jsonify({"error": "Missing question"}), 400
    if not session_id:
        return jsonify({"error": "Missing session ID"}), 400

    try:
        # Get session contexts
        with sqlite3.connect(DB_FILE) as db:
            rows = db.execute(
                "SELECT context_id, content FROM session_contexts WHERE session_id = ?",
                (session_id,),
            ).fetchall()
        session_contexts = [{"id": row[0], "content": row[1]} for row in rows]

        # Get relevant contexts
        relevant_contexts = get_relevant_contexts(
            question, session_contexts, session_id, api_key, model
        )

        # Prepare context for GPT
        context_text = "\n\n".join([ctx["content"] for ctx in relevant_contexts[:3]])
        prompt = f"Context:\n{context_text}\n\nQuestion: {question}\nAnswer:"

        # Call GPT (simplified)
        client = OpenAI(api_key=api_key)
        response = client.completions.create(
            model="gpt-3.5-turbo-instruct",
            prompt=prompt,
            max_tokens=500,
            temperature=0.3,
        )
        answer = response.choices[0].text.strip()

        # Store conversation
        with sqlite3.connect(DB_FILE) as db:
            db.execute(
                "INSERT INTO conversation_history (session_id, question, answer) VALUES (?, ?, ?)",
                (session_id, question, answer),
            )
            db.commit()

        # Generate follow-up questions
        follow_up_prompt = (
            f"Given this question: '{question}' and this answer: '{answer}', "
            "generate 3 relevant follow-up questions. Output as JSON array."
        )
        follow_up_response = client.completions.create(
            model="gpt-3.5-turbo-instruct",
            prompt=follow_up_prompt,
            max_tokens=150,
            temperature=0.5,
        )
        follow_ups = json.loads(follow_up_response.choices[0].text.strip())

        return jsonify(
            {
                "question": question,
                "answer": answer,
                "contexts": [ctx["content"] for ctx in relevant_contexts],
                "follow_ups": follow_ups,
            }
        )
    except Exception as e:
        logging.error(f"Question answering failed: {str(e)}")
        return jsonify({"error": str(e)}), 500


# Main page
@app.route("/")
def index():
    return render_template("kaggleloc.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
