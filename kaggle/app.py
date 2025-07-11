import os
import logging
import hashlib
import tempfile
import shutil
import zipfile
import json
import base64
from datetime import datetime
from pathlib import Path
from flask import Flask, request, jsonify, render_template, send_file
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

# Config
UPLOAD_FOLDER = "uploads"
CONVERSATIONS_FILE = "conversations.json"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CrossbarApp")

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
ADMIN_CREDENTIALS = {
    "username": os.environ.get("ADMIN_USER", "admin"),
    "password": os.environ.get("ADMIN_PASSWORD", "admin"),
}

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Utilities


def allowed_file(filename):
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def hash_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def extract_text_from_file(filepath):
    ext = Path(filepath).suffix.lower()
    text = ""
    try:
        if ext == ".pdf":
            with pdfplumber.open(filepath) as pdf:
                text = "\n".join(
                    page.extract_text() for page in pdf.pages if page.extract_text()
                )
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
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read()
    except Exception as e:
        logger.error(f"[extract_text] {filepath} failed: {e}")
    return text[:20000]


def is_image_heavy_pdf(filepath):
    try:
        doc = fitz.open(filepath)
        return sum(1 for p in doc if len(p.get_images(full=True)) > 0) / len(doc) > 0.5
    except:
        return False


def embed(text, api_key):
    client = OpenAI(api_key=api_key)
    return (
        client.embeddings.create(
            model="text-embedding-3-large", input=[text], encoding_format="float"
        )
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


# Routes
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
def reset_db():
    shutil.rmtree(UPLOAD_FOLDER, ignore_errors=True)
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    if os.path.exists(CONVERSATIONS_FILE):
        os.remove(CONVERSATIONS_FILE)
    return jsonify({"status": "reset complete"})


@app.route("/api/ask", methods=["POST"])
def ask():
    data = request.json
    question = data.get("question")
    session_id = data.get("sessionId")
    api_key = request.headers.get("Authorization", "").replace("Bearer ", "")
    if not question or not api_key:
        return jsonify({"error": "Missing question or API key"}), 400
    answer = f"Answer to: {question}"  # Replace with real RAG logic
    save_conversation(session_id, question, answer)
    return jsonify(
        {
            "answer": answer,
            "question": question,
            "follow_ups": ["Can you elaborate?", "Why does this matter?"],
        }
    )


@app.route("/api/download_conversation", methods=["GET"])
def download():
    fmt = request.args.get("format", "txt")
    session_id = request.args.get("sessionId")
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


@app.route("/api/upload_context", methods=["POST"])
def upload_context():
    api_key = request.headers.get("Authorization", "").replace("Bearer ", "")
    model = request.form.get("model", "text-embedding-3-large")
    session_id = request.form.get("sessionId")
    results = []
    seen_hashes = set()
    if "text" in request.form:
        text = request.form["text"]
        h = hash_text(text)
        if h not in seen_hashes:
            seen_hashes.add(h)
            embedding = embed(text, api_key)
            concepts = extract_concepts(text, api_key)
            results.append({"text": text, "concepts": concepts.split(", ")})
    elif "filesBase64" in request.form:
        files = request.form.getlist("filesBase64")
        filenames = request.form.getlist("filenames")
        for i, b64 in enumerate(files):
            _, encoded = b64.split(",", 1)
            file_bytes = base64.b64decode(encoded)
            path = os.path.join(UPLOAD_FOLDER, secure_filename(filenames[i]))
            with open(path, "wb") as f:
                f.write(file_bytes)
            text = extract_text_from_file(path)
            h = hash_text(text)
            if h in seen_hashes:
                continue
            seen_hashes.add(h)
            embedding = embed(text, api_key)
            concepts = extract_concepts(text, api_key)
            results.append({"filename": filenames[i], "concepts": concepts.split(", ")})
    return jsonify({"status": "success", "concepts": [r["concepts"] for r in results]})


@app.route("/")
def index():
    return render_template("kaggleloc.html")


if __name__ == "__main__":
    app.run(debug=True)
