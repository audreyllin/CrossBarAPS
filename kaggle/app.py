import os
import logging
import hashlib
import json
import base64
import subprocess
import shutil
import tempfile
from datetime import datetime
from functools import wraps
from pathlib import Path
from flask import (
    Flask,
    request,
    jsonify,
    render_template,
    send_file,
    redirect,
    url_for,
    current_app,
)
from werkzeug.utils import secure_filename
from openai import OpenAI
from fpdf import FPDF
from docx import Document as DocxWriter
import pytesseract
from PIL import Image
import pdfplumber
import fitz
from docx import Document as DocxDoc
from pptx import Presentation
import pandas as pd
import numpy as np
import faiss
from collections import Counter
import zipfile
import magic
from rag_pipeline import generate_media

# Configuration
UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "output"
CONVERSATIONS_FILE = "conversations.json"
CONTEXT_UPLOADS_FILE = "context_uploads.json"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DocumentAIApp")

# Supported file types
ALLOWED_EXTENSIONS = {
    ".txt",
    ".pdf",
    ".doc",
    ".docx",
    ".ppt",
    ".pptx",
    ".xls",
    ".xlsx",
    ".csv",
    ".tsv",
    ".jpg",
    ".jpeg",
    ".png",
    ".odt",
    ".rtf",
    ".zip",
}

# Admin credentials
ADMIN_CREDENTIALS = {
    "username": os.environ.get("ADMIN_USER", "admin"),
    "password": os.environ.get("ADMIN_PASSWORD", "admin"),
}

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024  # 100MB limit


# FAISS Vector Index
class VectorIndex:
    def __init__(self, dim=3072):
        self.index = faiss.IndexIDMap(faiss.IndexFlatL2(dim))
        self.metadata = {}
        self.next_id = 1

    def add_vector(self, vector, metadata):
        vector_id = self.next_id
        self.index.add_with_ids(np.array([vector]), np.array([vector_id]))
        self.metadata[vector_id] = metadata
        self.next_id += 1
        return vector_id

    def remove_vector(self, vector_id):
        self.index.remove_ids(np.array([vector_id]))
        if vector_id in self.metadata:
            del self.metadata[vector_id]

    def search(self, vector, k=5):
        distances, vector_ids = self.index.search(np.array([vector]), k)
        results = []
        for i in range(len(vector_ids[0])):
            if vector_ids[0][i] >= 0 and vector_ids[0][i] in self.metadata:
                results.append(
                    {
                        **self.metadata[vector_ids[0][i]],
                        "similarity": 1 - distances[0][i],
                    }
                )
        return results


# Initialize index
vector_index = VectorIndex()


# Load existing context
def load_context_index():
    global vector_index
    if os.path.exists(CONTEXT_UPLOADS_FILE):
        try:
            with open(CONTEXT_UPLOADS_FILE, "r") as f:
                context_data = json.load(f)
                for entry in context_data:
                    if "embedding" in entry:
                        vector_index.add_vector(
                            np.array(entry["embedding"]),
                            {
                                "context_id": entry["context_id"],
                                "session_id": entry["session_id"],
                                "filename": entry.get("filename"),
                                "text": entry.get("text"),
                                "concepts": entry.get("concepts", []),
                                "priority": entry.get("priority", 0),
                                "timestamp": entry["timestamp"],
                            },
                        )
        except Exception as e:
            logger.error(f"Error loading context index: {str(e)}")


# Save context to JSON
def save_context_upload(entry):
    data = []
    if os.path.exists(CONTEXT_UPLOADS_FILE):
        with open(CONTEXT_UPLOADS_FILE, "r") as f:
            try:
                data = json.load(f)
            except:
                pass

    entry_copy = entry.copy()
    if "embedding" in entry_copy:
        del entry_copy["embedding"]
    data.append(entry_copy)

    with open(CONTEXT_UPLOADS_FILE, "w") as f:
        json.dump(data, f, indent=2)


# Utility functions
def allowed_file(filename):
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def hash_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def is_image_heavy_pdf(filepath):
    try:
        doc = fitz.open(filepath)
        image_page_count = sum(1 for p in doc if len(p.get_images(full=True)) > 0)
        return image_page_count / len(doc) > 0.5
    except:
        return False


def extract_text_from_zip(zip_path):
    extracted_text = ""
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        for file in zip_ref.namelist():
            if any(file.lower().endswith(ext) for ext in ALLOWED_EXTENSIONS):
                with zip_ref.open(file) as f:
                    try:
                        # Save to temp file for processing
                        with tempfile.NamedTemporaryFile(delete=False) as tmp:
                            tmp.write(f.read())
                            tmp_path = tmp.name

                        extracted_text += extract_text_from_file(tmp_path) + "\n\n"
                        os.unlink(tmp_path)
                    except Exception as e:
                        logger.error(f"Error processing {file} in ZIP: {str(e)}")
    return extracted_text


def extract_text_from_file(filepath, api_key=None):
    ext = Path(filepath).suffix.lower()
    text = ""

    try:
        # Handle ZIP files
        if ext == ".zip":
            return extract_text_from_zip(filepath)

        # Handle other file types
        if ext == ".pdf":
            # First try regular text extraction
            with pdfplumber.open(filepath) as pdf:
                text = "\n".join(
                    page.extract_text() for page in pdf.pages if page.extract_text()
                )

            # If no text or image-heavy, try OCR
            if not text.strip() or is_image_heavy_pdf(filepath):
                doc = fitz.open(filepath)
                for page in doc:
                    pix = page.get_pixmap()
                    with tempfile.NamedTemporaryFile(
                        suffix=".png", delete=False
                    ) as tmp_img:
                        pix.save(tmp_img.name)
                        image = Image.open(tmp_img.name)
                        text += pytesseract.image_to_string(image) + "\n"
                        os.unlink(tmp_img.name)

        elif ext in [".doc", ".docx"]:
            doc = DocxDoc(filepath)
            text = "\n".join(p.text for p in doc.paragraphs)

        elif ext in [".ppt", ".pptx"]:
            prs = Presentation(filepath)
            text = "\n".join(
                shape.text
                for slide in prs.slides
                for shape in slide.shapes
                if hasattr(shape, "text")
            )

        elif ext in [".xls", ".xlsx"]:
            df = pd.read_excel(filepath, sheet_name=None)
            text = "\n".join(
                df[sheet].astype(str).apply(" | ".join, axis=1).str.cat(sep="\n")
                for sheet in df
            )

        elif ext in [".csv", ".tsv"]:
            text = (
                pd.read_csv(filepath)
                .astype(str)
                .apply(" | ".join, axis=1)
                .str.cat(sep="\n")
            )

        elif ext in [".jpg", ".jpeg", ".png"]:
            text = pytesseract.image_to_string(Image.open(filepath))

        elif ext == ".txt":
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()

    except Exception as e:
        logger.error(f"[extract_text] {filepath} failed: {str(e)}")

    return text[:20000]  # Limit to 20k characters


def embed(text, api_key, model="text-embedding-3-large"):
    client = OpenAI(api_key=api_key)
    return (
        client.embeddings.create(model=model, input=[text], encoding_format="float")
        .data[0]
        .embedding
    )


def extract_concepts(text, api_key):
    client = OpenAI(api_key=api_key)
    prompt = (
        "Extract key concepts and summarize the main ideas from the following text:\n"
        + text[:4000]
    )
    return (
        client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
        )
        .choices[0]
        .message.content
    )


def save_conversation(session_id, question, answer):
    history = []
    if os.path.exists(CONVERSATIONS_FILE):
        with open(CONVERSATIONS_FILE, "r", encoding="utf-8") as f:
            history = json.load(f)

    history.append(
        {
            "session_id": session_id,
            "question": question,
            "answer": answer,
            "timestamp": datetime.utcnow().isoformat(),
        }
    )

    with open(CONVERSATIONS_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)


def load_conversation():
    if os.path.exists(CONVERSATIONS_FILE):
        with open(CONVERSATIONS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def save_context_upload_with_priority(
    session_id, filename=None, text=None, concepts=[], priority=0
):
    context_id = hashlib.md5(
        f"{session_id}{datetime.utcnow().isoformat()}".encode()
    ).hexdigest()

    entry = {
        "context_id": context_id,
        "session_id": session_id,
        "timestamp": datetime.utcnow().isoformat(),
        "concepts": concepts,
        "priority": priority,
    }

    if filename:
        entry["filename"] = filename
        entry["type"] = "file"
    elif text:
        entry["text"] = text[:100] + "..." if len(text) > 100 else text
        entry["type"] = "text"

    save_context_upload(entry)
    return context_id


def trigger_index_rebuild():
    """Delete existing index to trigger rebuild on next question"""
    index_path = os.path.join(OUTPUT_FOLDER, "index.faiss")
    metadata_path = os.path.join(OUTPUT_FOLDER, "metadata.json")

    try:
        if os.path.exists(index_path):
            os.remove(index_path)
        if os.path.exists(metadata_path):
            os.remove(metadata_path)
        logger.info("Index files removed, will rebuild on next question")
    except Exception as e:
        logger.error(f"Error removing index files: {str(e)}")


# Authentication decorator
def requires_admin_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth = request.authorization
        if not auth or not (
            auth.username == ADMIN_CREDENTIALS["username"]
            and auth.password == ADMIN_CREDENTIALS["password"]
        ):
            return jsonify({"error": "Unauthorized"}), 401
        return f(*args, **kwargs)

    return decorated


# API Routes
@app.route("/api/admin_login", methods=["POST"])
def admin_login():
    data = request.json
    if (
        data.get("username") == ADMIN_CREDENTIALS["username"]
        and data.get("password") == ADMIN_CREDENTIALS["password"]
    ):
        return jsonify({"status": "success"})
    return jsonify({"error": "Unauthorized"}), 403


@app.route("/api/admin/reset_db", methods=["POST"])
@requires_admin_auth
def reset_db():
    try:
        shutil.rmtree(UPLOAD_FOLDER, ignore_errors=True)
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
        shutil.rmtree(OUTPUT_FOLDER, ignore_errors=True)
        os.makedirs(OUTPUT_FOLDER, exist_ok=True)

        if os.path.exists(CONVERSATIONS_FILE):
            os.remove(CONVERSATIONS_FILE)
        if os.path.exists(CONTEXT_UPLOADS_FILE):
            os.remove(CONTEXT_UPLOADS_FILE)

        # Reset vector index
        vector_index.index.reset()
        vector_index.metadata = {}
        vector_index.next_id = 1

        return jsonify({"status": "reset complete"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/ask", methods=["POST"])
def api_ask():
    data = request.json
    question = data.get("question")
    session_id = data.get("sessionId")
    model = data.get("model", "text-embedding-3-large")
    api_key = request.headers.get("Authorization", "").replace("Bearer ", "")

    if not question or not api_key:
        return jsonify({"error": "Missing question or API key"}), 400

    try:
        # Compute question embedding
        q_emb = embed(question, api_key, model)

        # Retrieve relevant context
        results = vector_index.search(q_emb, k=5)

        # Sort by priority then similarity
        results.sort(key=lambda x: (-x.get("priority", 0), -x["similarity"]))

        # Build context string
        context_str = ""
        for res in results:
            if res.get("text"):
                context_str += f"{res['text']}\n\n"
            elif res.get("filename"):
                context_str += f"From {res['filename']}:\n{res.get('text', '')}\n\n"

        # Build prompt with context
        prompt = f"""
        Context information:
        {context_str}
        
        Question: {question}
        Answer:
        """

        # Send to GPT
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
        )
        answer = response.choices[0].message.content.strip()

        # Save conversation
        save_conversation(session_id, question, answer)

        # Generate follow-up questions
        follow_ups = [
            "Can you elaborate on this?",
            "What are the key points from this answer?",
            "How does this relate to other documents?",
        ]

        return jsonify(
            {
                "answer": answer,
                "question": question,
                "follow_ups": follow_ups,
                "context_used": [r.get("filename") or "text snippet" for r in results],
            }
        )

    except Exception as e:
        logger.error(f"Error in api_ask: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/upload", methods=["POST"])
def upload_file():
    if "files" not in request.files:
        return jsonify({"error": "No files provided"}), 400

    files = request.files.getlist("files")
    session_id = request.form.get("sessionId")
    api_key = request.form.get("apiKey")
    model = request.form.get("model", "text-embedding-3-large")

    if not api_key:
        return jsonify({"error": "API key required"}), 400

    results = []
    seen_hashes = set()

    for file in files:
        if file.filename == "":
            continue

        if not allowed_file(file.filename):
            continue

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        try:
            # Extract text
            text = extract_text_from_file(filepath)
            if not text.strip():
                continue

            h = hash_text(text)
            if h in seen_hashes:
                continue
            seen_hashes.add(h)

            # Embed and extract concepts
            embedding = embed(text, api_key, model)
            concepts = extract_concepts(text, api_key).split(", ")

            # Add to vector index
            metadata = {
                "context_id": hashlib.md5(
                    f"{session_id}{datetime.utcnow().isoformat()}".encode()
                ).hexdigest(),
                "session_id": session_id,
                "filename": filename,
                "text": text,
                "concepts": concepts,
                "priority": 1,  # File context has higher priority
                "timestamp": datetime.utcnow().isoformat(),
            }
            vector_id = vector_index.add_vector(embedding, metadata)

            # Save to context uploads
            entry = {
                **metadata,
                "embedding": embedding.tolist(),
            }
            save_context_upload(entry)

            results.append(
                {
                    "filename": filename,
                    "status": "processed",
                    "concepts": concepts,
                }
            )

        except Exception as e:
            logger.error(f"Error processing {filename}: {str(e)}")
            results.append(
                {
                    "filename": filename,
                    "status": "error",
                    "error": str(e),
                }
            )
            continue

    trigger_index_rebuild()
    return jsonify({"status": "success", "results": results})


@app.route("/api/list_contexts", methods=["GET"])
def list_contexts():
    session_id = request.args.get("sessionId")
    if not session_id:
        return jsonify({"error": "Missing sessionId"}), 400

    contexts = []
    if os.path.exists(CONTEXT_UPLOADS_FILE):
        try:
            with open(CONTEXT_UPLOADS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                contexts = [
                    entry for entry in data if entry.get("session_id") == session_id
                ]
        except Exception as e:
            logger.error(f"Error loading context uploads: {str(e)}")

    return jsonify({"status": "success", "contexts": contexts})


@app.route("/api/upload_context", methods=["POST"])
def upload_context():
    api_key = request.headers.get("Authorization", "").replace("Bearer ", "")
    if not api_key:
        return jsonify({"error": "Missing API key"}), 401

    model = request.form.get("model", "text-embedding-3-large")
    session_id = request.form.get("sessionId")
    if not session_id:
        return jsonify({"error": "Missing sessionId"}), 400

    results = []
    seen_hashes = set()

    if "text" in request.form:
        text = request.form["text"]
        h = hash_text(text)
        if h not in seen_hashes:
            seen_hashes.add(h)
            try:
                embedding = embed(text, api_key, model)
                concepts = extract_concepts(text, api_key).split(", ")

                # Add to vector index
                metadata = {
                    "context_id": hashlib.md5(
                        f"{session_id}{datetime.utcnow().isoformat()}".encode()
                    ).hexdigest(),
                    "session_id": session_id,
                    "filename": None,
                    "text": text,
                    "concepts": concepts,
                    "priority": 0,  # Text context has lower priority
                    "timestamp": datetime.utcnow().isoformat(),
                }
                vector_id = vector_index.add_vector(embedding, metadata)

                # Save to context uploads
                entry = {
                    **metadata,
                    "embedding": embedding.tolist(),
                }
                save_context_upload(entry)

                results.append(
                    {
                        "context_id": metadata["context_id"],
                        "text": text,
                        "concepts": concepts,
                    }
                )

            except Exception as e:
                return jsonify({"error": str(e)}), 500

    elif "filesBase64" in request.form:
        files = request.form.getlist("filesBase64")
        filenames = request.form.getlist("filenames")

        for i, b64 in enumerate(files):
            try:
                _, encoded = b64.split(",", 1)
                file_bytes = base64.b64decode(encoded)
                filename = secure_filename(filenames[i])
                path = os.path.join(UPLOAD_FOLDER, filename)

                with open(path, "wb") as f:
                    f.write(file_bytes)

                # Extract text
                text = extract_text_from_file(path)
                h = hash_text(text)
                if h in seen_hashes:
                    continue
                seen_hashes.add(h)

                # Embed and extract concepts
                embedding = embed(text, api_key, model)
                concepts = extract_concepts(text, api_key).split(", ")

                # Add to vector index
                metadata = {
                    "context_id": hashlib.md5(
                        f"{session_id}{datetime.utcnow().isoformat()}".encode()
                    ).hexdigest(),
                    "session_id": session_id,
                    "filename": filename,
                    "text": text,
                    "concepts": concepts,
                    "priority": 1,  # File context has higher priority
                    "timestamp": datetime.utcnow().isoformat(),
                }
                vector_id = vector_index.add_vector(embedding, metadata)

                # Save to context uploads
                entry = {
                    **metadata,
                    "embedding": embedding.tolist(),
                }
                save_context_upload(entry)

                results.append(
                    {
                        "context_id": metadata["context_id"],
                        "filename": filename,
                        "concepts": concepts,
                    }
                )

            except Exception as e:
                logger.error(f"Error processing file {filenames[i]}: {str(e)}")
                continue

    return jsonify({"status": "success", "results": results})


@app.route("/api/remove_context", methods=["POST"])
def remove_context():
    data = request.json
    context_id = data.get("context_id")
    session_id = data.get("session_id")
    api_key = request.headers.get("Authorization", "").replace("Bearer ", "")

    if not context_id or not session_id:
        return jsonify({"error": "Missing context_id or session_id"}), 400

    # Remove from vector index
    vector_id_to_remove = None
    for vec_id, meta in vector_index.metadata.items():
        if meta["context_id"] == context_id and meta["session_id"] == session_id:
            vector_id_to_remove = vec_id
            break

    if vector_id_to_remove:
        vector_index.remove_vector(vector_id_to_remove)

    # Update context_uploads.json
    if os.path.exists(CONTEXT_UPLOADS_FILE):
        try:
            with open(CONTEXT_UPLOADS_FILE, "r", encoding="utf-8") as f:
                context_data = json.load(f)
            context_data = [e for e in context_data if e["context_id"] != context_id]
            with open(CONTEXT_UPLOADS_FILE, "w", encoding="utf-8") as f:
                json.dump(context_data, f, indent=2)
        except Exception as e:
            logger.error(f"Error removing context from file: {e}")

    return jsonify({"status": "success"})


@app.route("/api/download_conversation", methods=["GET"])
def download():
    fmt = request.args.get("format", "txt")
    session_id = request.args.get("sessionId")

    if not session_id:
        return jsonify({"error": "Missing sessionId"}), 400

    convo = [c for c in load_conversation() if c["session_id"] == session_id]
    if not convo:
        return jsonify({"error": "No data"}), 404

    content = "\n\n".join([f"Q: {c['question']}\nA: {c['answer']}" for c in convo])

    if fmt == "txt":
        return content, 200, {"Content-Type": "text/plain"}
    elif fmt == "pdf":
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        for line in content.splitlines():
            pdf.cell(200, 10, txt=line, ln=1)
        pdf_path = "conversation.pdf"
        pdf.output(pdf_path)
        return send_file(pdf_path, as_attachment=True)
    elif fmt == "docx":
        doc = DocxWriter()
        for q in convo:
            doc.add_paragraph(f"Q: {q['question']}")
            doc.add_paragraph(f"A: {q['answer']}\n")
        doc_path = "conversation.docx"
        doc.save(doc_path)
        return send_file(doc_path, as_attachment=True)
    else:
        return jsonify({"error": "invalid format"}), 400


# Admin routes
@app.route("/api/admin/uploads", methods=["GET"])
@requires_admin_auth
def admin_uploads():
    if os.path.exists(CONTEXT_UPLOADS_FILE):
        with open(CONTEXT_UPLOADS_FILE, "r") as f:
            return jsonify(json.load(f))
    return jsonify([])


@app.route("/api/admin/conversations", methods=["GET"])
@requires_admin_auth
def admin_conversations():
    if os.path.exists(CONVERSATIONS_FILE):
        with open(CONVERSATIONS_FILE, "r") as f:
            return jsonify(json.load(f))
    return jsonify([])


@app.route("/api/admin/frequent_questions", methods=["GET"])
@requires_admin_auth
def frequent_questions():
    n = int(request.args.get("n", 10))
    if os.path.exists(CONVERSATIONS_FILE):
        with open(CONVERSATIONS_FILE, "r") as f:
            conversations = json.load(f)
            questions = [c["question"] for c in conversations]
            counter = Counter(questions)
            top_n = counter.most_common(n)
            return jsonify([{"question": q, "count": c} for q, c in top_n])
    return jsonify([])


@app.route("/api/generate", methods=["POST"])
def api_generate():
    data = request.get_json() or {}
    media_type = data.get("type")
    answer = data.get("answer", "")
    session_id = data.get("sessionId")
    api_key = request.headers.get("Authorization", "").replace("Bearer ", "")

    if not api_key:
        return jsonify({"error": "Missing API key"}), 401
    if media_type not in ("video", "poster", "slides", "memo"):
        return jsonify({"error": f"Invalid type '{media_type}'"}), 400

    try:
        output_path = generate_media(media_type, answer, session_id, api_key)

        # SECOND SAFETY CHECK
        if not output_path or not os.path.isfile(output_path):
            current_app.logger.error(
                "generate_media returned invalid path: %r", output_path
            )
            return (
                jsonify({"error": "Generation succeeded but no file was created"}),
                500,
            )

        filename = os.path.basename(output_path)
        # Build a fully‐qualified URL to your download endpoint
        download_url = url_for("download_generated", file=filename, _external=True)
        return jsonify({"url": download_url})

    except Exception as e:
        current_app.logger.exception("Error generating media")
        return jsonify({"error": str(e)}), 500


@app.route("/api/download_generated", methods=["GET"])
def download_generated():
    file_name = request.args.get("file")
    if not file_name:
        return jsonify({"error": "Missing file"}), 400

    fp = os.path.join(OUTPUT_FOLDER, file_name)
    if not os.path.exists(fp):
        return jsonify({"error": "Not found"}), 404

    return send_file(fp, as_attachment=True)


# Main route
@app.route("/")
def index():
    return render_template("kaggle.html")


if __name__ == "__main__":
    load_context_index()
    app.run(host="0.0.0.0", port=5000, debug=True)
