import os
import logging
import uuid
import tiktoken
import json
import hashlib
import re
import sqlite3
import numpy as np
import base64
import tempfile
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
from werkzeug.utils import secure_filename
from collections import Counter
from dotenv import load_dotenv
from rag_pipeline import (
    extract_text_from_file,
    extract_with_gpt_vision_base64,
    extract_image_metadata,
    is_image_heavy_pdf,
)
from flask import Flask, request, jsonify, render_template, send_file
from io import BytesIO
from docx import Document as DocxDoc
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

# Load environment variables
load_dotenv()

# Initialize app and paths
current_dir = os.path.dirname(os.path.abspath(__file__))
template_path = os.path.join(current_dir, "templates")
DB_FILE = os.path.join(current_dir, "memory.db")
UPLOAD_FOLDER = "uploads"
EMBEDDED_FLAGS = "embedded_flags"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(EMBEDDED_FLAGS, exist_ok=True)
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

# Admin credentials
ADMIN_USER = os.environ.get("ADMIN_USER", "admin")
ADMIN_PASS = os.environ.get("ADMIN_PASS", "securepassword")

app = Flask(__name__, template_folder=template_path)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
logging.basicConfig(level=logging.INFO)

# Constants
IMAGE_RATIO_THRESHOLD = 0.5  # >50% images considered image-heavy


# Initialize database
def init_db():
    with sqlite3.connect(DB_FILE) as db:
        # Create tables if they don't exist
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
                content_hash TEXT NOT NULL,
                source_type TEXT,
                filename TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                is_active BOOLEAN DEFAULT 1
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
        db.execute(
            """
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                timestamp TEXT,
                question TEXT,
                answer TEXT,
                file TEXT
            )
            """
        )
        db.execute(
            """
            CREATE TABLE IF NOT EXISTS uploads (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                filename TEXT,
                timestamp TEXT
            )
            """
        )
        # Create indexes
        db.execute(
            "CREATE INDEX IF NOT EXISTS idx_content_hash ON session_contexts (content_hash)"
        )
        db.execute(
            "CREATE INDEX IF NOT EXISTS idx_session_active ON session_contexts (session_id, is_active)"
        )


init_db()

# Token counting
encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")


def count_tokens(text):
    return len(encoding.encode(text))


# GPT Embedding
def generate_vector(text, api_key, model="text-embedding-3-large"):
    try:
        client = OpenAI(api_key=api_key)
        response = client.embeddings.create(
            model=model, input=[text[:8192]]  # Truncate to model limit
        )
        return np.array(response.data[0].embedding)
    except Exception as e:
        logging.error(f"Embedding generation failed: {str(e)}")
        dim = 3072 if model == "text-embedding-3-large" else 1536
        return np.zeros(dim)


# Concept extraction with GPT
def extract_concepts(text, api_key):
    try:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "Extract top 5 key concepts as JSON array",
                },
                {"role": "user", "content": f"Extract concepts from:\n\n{text[:3000]}"},
            ],
            response_format={"type": "json_object"},
            max_tokens=150,
            temperature=0.3,
        )
        response_content = response.choices[0].message.content
        concepts = json.loads(response_content).get("concepts", [])
        if isinstance(concepts, list) and len(concepts) > 0:
            return [concept.capitalize() for concept in concepts]
    except Exception as e:
        logging.error(f"Concept extraction failed: {str(e)}")

    # Fallback to GPT-3.5 if GPT-4 fails
    try:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "Extract top 5 key concepts as comma-separated list",
                },
                {"role": "user", "content": f"Extract concepts from:\n\n{text[:3000]}"},
            ],
            max_tokens=100,
            temperature=0.3,
        )
        concepts = response.choices[0].message.content.strip().split(",")
        return [concept.strip().capitalize() for concept in concepts[:5]]
    except Exception as e:
        logging.error(f"Fallback concept extraction failed: {str(e)}")
        return ["General", "Information", "Technology", "Business", "Data"]


# Context retrieval
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
            logging.error(f"Similarity error: {str(e)}")
            similarities.append((context_id, 0))

    similarities.sort(key=lambda x: x[1], reverse=True)
    top_contexts = [ctx_id for ctx_id, _ in similarities[:top_n]]
    return [ctx for ctx in session_contexts if ctx["id"] in top_contexts]


# File processing functions
def process_all_uploads_on_start():
    print("[Startup] Processing files in uploads/ folder...")
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logging.warning("OPENAI_API_KEY not set. Skipping file auto-processing.")
        return

    for filename in os.listdir(UPLOAD_FOLDER):
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        if not os.path.isfile(filepath) or not allowed_file(filename):
            continue

        # Skip already-processed files
        if check_if_file_embedded(filename):
            print(f"[Skipped] {filename} already embedded.")
            continue

        try:
            print(f"[Processing] Embedding {filename}...")
            embed_and_store_file(filepath, api_key)
            mark_file_as_embedded(filename)
            print(f"[Done] Embedded {filename}.")
        except Exception as e:
            print(f"[Error] Failed to embed {filename}: {e}")


def check_if_file_embedded(filename):
    return os.path.exists(f"{EMBEDDED_FLAGS}/{filename}.done")


def mark_file_as_embedded(filename):
    with open(f"{EMBEDDED_FLAGS}/{filename}.done", "w") as f:
        f.write("embedded")


# ===== ADMIN ROUTES =====
@app.route("/api/admin_login", methods=["POST"])
def admin_login():
    data = request.get_json()
    if data.get("username") == ADMIN_USER and data.get("password") == ADMIN_PASS:
        return jsonify({"status": "success"})
    return jsonify({"error": "Invalid credentials"}), 401


@app.route("/api/admin/reset_db", methods=["POST"])
def reset_db():
    try:
        os.system("python init_db.py")
        return jsonify({"status": "reset successful"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/admin/conversations", methods=["GET"])
def admin_conversations():
    with sqlite3.connect(DB_FILE) as db:
        rows = db.execute(
            "SELECT id, session_id, timestamp, question, answer, file FROM conversations ORDER BY timestamp DESC"
        ).fetchall()
    return jsonify(
        [
            {
                "id": r[0],
                "session_id": r[1],
                "timestamp": r[2],
                "question": r[3],
                "answer": r[4],
                "file": r[5],
            }
            for r in rows
        ]
    )


@app.route("/api/admin/frequent_questions", methods=["GET"])
def frequent_questions():
    with sqlite3.connect(DB_FILE) as db:
        rows = db.execute(
            "SELECT question, COUNT(*) as count FROM conversations GROUP BY question ORDER BY count DESC LIMIT 10"
        ).fetchall()
    return jsonify([{"question": r[0], "count": r[1]} for r in rows])


@app.route("/api/admin/uploads", methods=["GET"])
def admin_uploads():
    with sqlite3.connect(DB_FILE) as db:
        rows = db.execute(
            "SELECT id, session_id, filename, timestamp FROM uploads ORDER BY timestamp DESC"
        ).fetchall()
    return jsonify(
        [
            {
                "id": r[0],
                "session_id": r[1],
                "filename": r[2],
                "timestamp": r[3],
            }
            for r in rows
        ]
    )


# ===== FILE MANAGEMENT ROUTES =====
@app.route("/api/download_conversation", methods=["GET"])
def download_convo():
    session_id = request.args.get("sessionId")
    fmt = request.args.get("format", "txt")
    if not session_id:
        return jsonify({"error": "Missing session ID"}), 400

    with sqlite3.connect(DB_FILE) as db:
        rows = db.execute(
            "SELECT question, answer, timestamp FROM conversation_history WHERE session_id = ?",
            (session_id,),
        ).fetchall()

    content = [f"Q: {q}\nA: {a}\nTime: {t}" for q, a, t in rows]

    if fmt == "txt":
        return (
            BytesIO("\n\n".join(content).encode("utf-8")),
            200,
            {
                "Content-Type": "text/plain",
                "Content-Disposition": f"attachment; filename=chat_{session_id}.txt",
            },
        )
    elif fmt == "docx":
        doc = DocxDoc()
        for entry in content:
            doc.add_paragraph(entry)
        buf = BytesIO()
        doc.save(buf)
        buf.seek(0)
        return send_file(
            buf, as_attachment=True, download_name=f"chat_{session_id}.docx"
        )
    elif fmt == "pdf":
        buf = BytesIO()
        c = canvas.Canvas(buf, pagesize=letter)
        width, height = letter
        y = height - 50
        for entry in content:
            for line in entry.split("\n"):
                c.drawString(50, y, line)
                y -= 15
                if y < 50:
                    c.showPage()
                    y = height - 50
        c.save()
        buf.seek(0)
        return send_file(
            buf, as_attachment=True, download_name=f"chat_{session_id}.pdf"
        )
    return jsonify({"error": "Unsupported format"}), 400


@app.route("/api/list_contexts", methods=["GET"])
def list_contexts():
    session_id = request.args.get("sessionId")
    if not session_id:
        return jsonify({"error": "Missing session ID"}), 400

    with sqlite3.connect(DB_FILE) as db:
        rows = db.execute(
            "SELECT context_id, filename, timestamp FROM session_contexts "
            "WHERE session_id = ? AND is_active = 1",
            (session_id,),
        ).fetchall()
    return jsonify(
        [{"context_id": r[0], "filename": r[1], "timestamp": r[2]} for r in rows]
    )


@app.route("/api/remove_context", methods=["POST"])
def remove_context():
    data = request.get_json()
    context_id = data.get("contextId")
    session_id = data.get("sessionId")
    if not context_id or not session_id:
        return jsonify({"error": "Missing parameters"}), 400

    with sqlite3.connect(DB_FILE) as db:
        db.execute(
            "UPDATE session_contexts SET is_active = 0 WHERE context_id = ? AND session_id = ?",
            (context_id, session_id),
        )
        db.execute(
            "DELETE FROM context_vectors WHERE context_id = ? AND session_id = ?",
            (context_id, session_id),
        )
        db.commit()
    return jsonify({"status": "removed"})


# ===== MAIN FUNCTIONALITY ROUTES =====
def process_text_content(
    text_content, api_key, session_id, model, filename, source_type
):
    content_hash = hashlib.sha256(text_content.encode()).hexdigest()

    # Check for duplicates
    with sqlite3.connect(DB_FILE) as db:
        existing = db.execute(
            "SELECT context_id FROM session_contexts "
            "WHERE session_id = ? AND content_hash = ? AND is_active = 1",
            (session_id, content_hash),
        ).fetchone()

        if existing:
            return {
                "status": "duplicate",
                "filename": filename,
                "context_id": existing[0],
            }

    context_id = (
        f"file_{hashlib.md5((filename + session_id).encode()).hexdigest()[:10]}"
    )
    vector = generate_vector(text_content, api_key, model=model)

    with sqlite3.connect(DB_FILE) as db:
        db.execute(
            "INSERT INTO context_vectors (context_id, session_id, vector, model) "
            "VALUES (?, ?, ?, ?)",
            (context_id, session_id, json.dumps(vector.tolist()), model),
        )
        db.execute(
            "INSERT INTO session_contexts (session_id, context_id, content, source_type, filename, content_hash) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                session_id,
                context_id,
                text_content,
                source_type,
                filename,
                content_hash,
            ),
        )
        db.commit()

    concepts = extract_concepts(text_content, api_key)
    return {
        "filename": filename,
        "status": "success",
        "context_id": context_id,
        "concepts": concepts,
    }


@app.route("/api/upload_context", methods=["POST"])
def upload_context():
    api_key = request.headers.get("Authorization", "").replace("Bearer ", "")
    if not api_key:
        return jsonify({"error": "Missing API key"}), 401

    session_id = request.form.get("sessionId")
    model = request.form.get("model", "text-embedding-3-large")
    results = []

    # Handle text context
    if "text" in request.form:
        text_content = request.form["text"].strip()[:20000]
        if not text_content:
            return jsonify({"error": "Empty text content"}), 400

        results.append(
            process_text_content(
                text_content, api_key, session_id, model, "text_context", "text"
            )
        )

    # Handle Base64 files
    elif "filesBase64" in request.form:
        base64_files = request.form.getlist("filesBase64")
        filenames = request.form.getlist("filenames")

        for i, b64_data in enumerate(base64_files):
            filename = filenames[i] if i < len(filenames) else f"file_{i}"

            try:
                # Extract base64 string (remove data URL prefix)
                header, encoded = b64_data.split(",", 1)
                file_data = base64.b64decode(encoded)

                # Create temp file
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=os.path.splitext(filename)[1]
                ) as tmp_file:
                    tmp_file.write(file_data)
                    tmp_path = tmp_file.name

                # Process file
                result = process_file(tmp_path, api_key, session_id, model, filename)
                results.append(result)

                # After successful file processing, log in uploads table
                if result.get("status") == "success":
                    with sqlite3.connect(DB_FILE) as db:
                        db.execute(
                            "INSERT INTO uploads (session_id, filename, timestamp) VALUES (?, ?, datetime('now'))",
                            (session_id, filename),
                        )
                        db.commit()

                # Clean up temp file
                os.unlink(tmp_path)

            except Exception as e:
                results.append(
                    {"filename": filename, "status": "error", "error": str(e)}
                )
    else:
        return jsonify({"error": "No valid context provided"}), 400

    return jsonify(results)


def process_file(filepath, api_key, session_id, model, filename):
    """Process a file with visual content analysis"""
    try:
        # Extract text with visual analysis
        text_content = extract_text_from_file(filepath)
        file_ext = os.path.splitext(filepath)[1].lower()

        # Analyze content type
        if not text_content.strip():
            # Visual content only
            logging.info(
                f"Visual content only: {filename}, switching to image analysis"
            )
            image_analysis = extract_image_metadata(filepath)
            return {
                "status": "visual_only",
                "filename": filename,
                "image_analysis": image_analysis,
            }

        # Calculate image ratio (for PDFs)
        image_ratio = 0
        if file_ext == ".pdf":
            image_ratio = 1.0 if is_image_heavy_pdf(filepath) else 0.0

        # Handle different content types
        if image_ratio > IMAGE_RATIO_THRESHOLD:
            # Image-heavy document
            image_analysis = extract_image_metadata(filepath)
            return {
                "status": "image_heavy",
                "filename": filename,
                "text": text_content,
                "image_insights": image_analysis,
            }
        elif image_ratio > 0:
            # Mixed content
            image_analysis = extract_image_metadata(filepath)
            return {
                "status": "mixed",
                "filename": filename,
                "text": text_content,
                "image_insights": image_analysis,
            }
        else:
            # Text-heavy document
            return process_text_content(
                text_content, api_key, session_id, model, filename, "file"
            )

    except Exception as e:
        logging.error(f"File processing failed: {str(e)}")
        return {"filename": filename, "status": "error", "error": str(e)}


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
        # Get active contexts
        with sqlite3.connect(DB_FILE) as db:
            rows = db.execute(
                "SELECT context_id, content FROM session_contexts "
                "WHERE session_id = ? AND is_active = 1",
                (session_id,),
            ).fetchall()
        session_contexts = [{"id": row[0], "content": row[1]} for row in rows]

        # Get relevant contexts
        relevant_contexts = get_relevant_contexts(
            question, session_contexts, session_id, api_key, model
        )
        file_str = ", ".join([ctx["id"] for ctx in relevant_contexts])

        # Prepare context for GPT
        context_text = "\n\n".join([ctx["content"] for ctx in relevant_contexts[:3]])
        prompt = f"Context:\n{context_text}\n\nQuestion: {question}\nAnswer:"

        # Call GPT
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "Answer based on provided context"},
                {"role": "user", "content": prompt},
            ],
            max_tokens=500,
            temperature=0.3,
        )
        answer = response.choices[0].message.content.strip()

        # Store conversation in both tables
        with sqlite3.connect(DB_FILE) as db:
            # Store in conversation_history (session-based)
            db.execute(
                "INSERT INTO conversation_history (session_id, question, answer) "
                "VALUES (?, ?, ?)",
                (session_id, question, answer),
            )

            # Also store in conversations (admin dashboard)
            db.execute(
                "INSERT INTO conversations (session_id, timestamp, question, answer, file) "
                "VALUES (?, datetime('now'), ?, ?, ?)",
                (session_id, question, answer, file_str),
            )
            db.commit()

        # Generate follow-up questions with GPT
        follow_up_prompt = (
            f"Given the following conversation:\n"
            f"Question: {question}\n"
            f"Answer: {answer}\n\n"
            "Generate 3 relevant follow-up questions as a JSON array of strings."
        )

        follow_up_response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that generates follow-up questions",
                },
                {"role": "user", "content": follow_up_prompt},
            ],
            response_format={"type": "json_object"},
            max_tokens=150,
            temperature=0.5,
        )

        try:
            follow_ups = json.loads(follow_up_response.choices[0].message.content).get(
                "questions", []
            )
        except:
            follow_ups = [
                "Can you elaborate on that?",
                "What are the key points I should remember?",
                "How does this relate to other topics?",
            ]

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


@app.route("/api/generate_content", methods=["POST"])
def generate_content():
    api_key = request.headers.get("Authorization", "").replace("Bearer ", "")
    if not api_key:
        return jsonify({"error": "Missing API key"}), 401

    data = request.get_json()
    content = data.get("content", "").strip()
    content_type = data.get("type", "memo")

    if not content:
        return jsonify({"error": "Missing content"}), 400

    try:
        client = OpenAI(api_key=api_key)

        # Prompt templates for different content types
        prompts = {
            "memo": "Generate a structured company memo or whitepaper based on this content.",
            "poster": "Design a compelling marketing poster description with a title, tagline, and layout elements.",
            "slides": "Create a JSON array of 5 slide titles and bullet points for a pitch deck presentation.",
            "video": "Generate a short storyboard with scenes (title + scene description) for video generation.",
        }

        prompt = prompts.get(content_type, "Summarize this content.")

        # Call GPT-4
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "You are a professional content generator for enterprise knowledge management.",
                },
                {"role": "user", "content": f"{prompt}\n\nContent:\n{content[:4000]}"},
            ],
            max_tokens=700,
            temperature=0.5,
        )

        generated = response.choices[0].message.content.strip()

        # Format slides as JSON if requested
        if content_type == "slides":
            try:
                # Try to parse as JSON
                slides = json.loads(generated)
                return jsonify({"generated": slides})
            except:
                # If not valid JSON, return as is
                return jsonify({"generated": generated})

        return jsonify({"generated": generated})

    except Exception as e:
        logging.error(f"Content generation failed: {str(e)}")
        return jsonify({"error": str(e)}), 500


# Helper functions
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def index():
    return render_template("kaggle.html")


if __name__ == "__main__":
    process_all_uploads_on_start()  # Auto-process files on startup
    app.run(host="0.0.0.0", port=5000, debug=True)
