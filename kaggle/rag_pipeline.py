import os
import logging
import hashlib
import json
import base64
from pathlib import Path
from datetime import datetime
import tempfile
import pytesseract
import pdfplumber
import fitz
from PIL import Image
from openai import OpenAI
from docx import Document as DocxDoc
from pptx import Presentation
import pandas as pd
from fpdf import FPDF
from docx import Document as DocxWriter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rag_pipeline")

UPLOAD_FOLDER = "uploads"
CONVERSATIONS_FILE = "conversations.json"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ========== Utility Functions ===========


def hash_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def is_image_heavy_pdf(filepath):
    try:
        doc = fitz.open(filepath)
        return sum(1 for p in doc if len(p.get_images(full=True)) > 0) / len(doc) > 0.5
    except:
        return False


def extract_with_gpt_vision_base64(image_path):
    try:
        with open(image_path, "rb") as img:
            b64_image = base64.b64encode(img.read()).decode("utf-8")
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Extract all visible text from this image.",
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{b64_image}"},
                        },
                    ],
                }
            ],
            max_tokens=1024,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"GPT Vision extraction failed: {str(e)}")
        return ""


def extract_with_gpt_vision_from_pdf(filepath):
    try:
        doc = fitz.open(filepath)
        combined_text = ""
        for page in doc:
            pix = page.get_pixmap()
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_img:
                pix.save(tmp_img.name)
                combined_text += extract_with_gpt_vision_base64(tmp_img.name) + "\n"
                os.unlink(tmp_img.name)
        return combined_text
    except Exception as e:
        logger.error(f"GPT Vision fallback failed: {str(e)}")
        return ""


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
                text = extract_with_gpt_vision_from_pdf(filepath)
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


def download_conversation(session_id, fmt="txt"):
    convo = [c for c in load_conversation() if c["session_id"] == session_id]
    content = "\n\n".join([f"Q: {c['question']}\nA: {c['answer']}" for c in convo])
    if fmt == "txt":
        return content
    elif fmt == "pdf":
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        for line in content.splitlines():
            pdf.cell(200, 10, txt=line, ln=1)
        pdf_path = f"conversation_{session_id}.pdf"
        pdf.output(pdf_path)
        return pdf_path
    elif fmt == "docx":
        doc = DocxWriter()
        for q in convo:
            doc.add_paragraph(f"Q: {q['question']}")
            doc.add_paragraph(f"A: {q['answer']}\n")
        doc_path = f"conversation_{session_id}.docx"
        doc.save(doc_path)
        return doc_path
    return None