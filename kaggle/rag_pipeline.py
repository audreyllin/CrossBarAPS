import os
import json
import base64
import subprocess
import logging
import re
import numpy as np
from datetime import datetime
from pathlib import Path
import requests

# Document processing imports
import pytesseract
import pdfplumber
import fitz
from PIL import Image
from docx import Document as DocxReader
from docx import Document as DocxWriter
from pptx import Presentation
import replicate
from openai import OpenAI

# AI/Vector imports
import faiss

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("optimized_rag")

# Configuration
UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "output"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

CHUNK_SIZE = 1500
CHUNK_OVERLAP = 300
EMBEDDING_MODEL = "text-embedding-3-large"
GPT_MODEL = "gpt-4-turbo"
MIN_CHUNK_LENGTH = 100
MAX_FILE_SIZE_MB = 10
REPLICATE_API_TOKEN = os.environ.get("REPLICATE_API_TOKEN")


def clean_text(text):
    """Clean and normalize text by removing extra whitespace"""
    return re.sub(r"\s+", " ", text).strip()


def is_image_heavy_pdf(filepath):
    """Check if PDF contains mostly images"""
    try:
        doc = fitz.open(filepath)
        image_count = sum(len(page.get_images(full=True)) for page in doc)
        return image_count / len(doc) > 0.5
    except Exception:
        return False


def extract_pdf_text(filepath):
    """Extract text from PDF, falling back to OCR if needed"""
    text = ""
    try:
        # First try regular text extraction
        with pdfplumber.open(filepath) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""

        # Fallback to OCR if needed
        if not text.strip() or is_image_heavy_pdf(filepath):
            logger.info(f"Using OCR for PDF: {filepath}")
            doc = fitz.open(filepath)
            for page in doc:
                text += page.get_text() or ""
    except Exception as e:
        logger.error(f"PDF extraction failed: {str(e)}")
    return clean_text(text)[:50000]  # Limit to 50k characters


def extract_docx_text(filepath):
    """Extract text from Word documents"""
    try:
        doc = DocxReader(filepath)
        return clean_text("\n".join(p.text for p in doc.paragraphs))
    except Exception as e:
        logger.error(f"DOCX extraction failed: {str(e)}")
        return ""


def extract_text(filepath):
    """Main text extraction function that handles different file types"""
    ext = Path(filepath).suffix.lower()

    # Skip large files
    if os.path.getsize(filepath) > MAX_FILE_SIZE_MB * 1024 * 1024:
        logger.warning(f"Skipping large file: {filepath}")
        return ""

    try:
        if ext == ".pdf":
            return extract_pdf_text(filepath)
        elif ext in [".doc", ".docx"]:
            return extract_docx_text(filepath)
        elif ext in [".jpg", ".jpeg", ".png"]:
            return pytesseract.image_to_string(Image.open(filepath))
        elif ext == ".txt":
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                return clean_text(f.read())[:50000]
        else:
            logger.warning(f"Unsupported file format: {ext}")
            return ""
    except Exception as e:
        logger.error(f"Extraction error for {filepath}: {str(e)}")
        return ""


def chunk_text(text):
    """Split text into chunks with overlap, respecting sentence boundaries"""
    if len(text) <= CHUNK_SIZE:
        return [text]

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))

        # Adjust to nearest sentence boundary if possible
        if end < len(text):
            boundary = max(
                text.rfind(".", start, end),
                text.rfind("?", start, end),
                text.rfind("!", start, end),
            )
            if boundary > start + MIN_CHUNK_LENGTH:
                end = boundary + 1

        chunks.append(text[start:end])
        start = end - CHUNK_OVERLAP

    return chunks


def get_embeddings(texts, api_key):
    """Generate embeddings using OpenAI's API"""
    client = OpenAI(api_key=api_key)
    try:
        response = client.embeddings.create(
            model=EMBEDDING_MODEL, input=texts, encoding_format="float"
        )
        return [data.embedding for data in response.data]
    except Exception as e:
        logger.error(f"Embedding generation failed: {str(e)}")
        return []


def create_index(api_key):
    """Create a FAISS index from documents in the upload folder"""
    index = faiss.IndexFlatL2(3072)  # Dimension for text-embedding-3-large
    metadata = []

    for filepath in Path(UPLOAD_FOLDER).glob("*"):
        if filepath.suffix.lower() not in [".pdf", ".doc", ".docx", ".txt"]:
            continue

        logger.info(f"Processing: {filepath.name}")
        text = extract_text(str(filepath))
        if not text:
            continue

        chunks = chunk_text(text)
        embeddings = get_embeddings(chunks, api_key)

        for i, emb in enumerate(embeddings):
            if len(chunks[i]) > MIN_CHUNK_LENGTH:
                # Fix dtype warning by ensuring float32
                index.add(np.array([emb]).astype("float32"))
                metadata.append(
                    {
                        "filename": filepath.name,
                        "chunk_index": i,
                        "content": chunks[i][:500],  # Store snippet for reference
                    }
                )

    return index, metadata


def search_index(index, metadata, query_embedding, k=5):
    """Search the FAISS index for relevant chunks"""
    distances, indices = index.search(np.array([query_embedding]).astype("float32"), k)
    return [metadata[i] for i in indices[0] if i < len(metadata)]


def generate_answer(question, context, api_key):
    """Generate an answer using GPT with provided context"""
    client = OpenAI(api_key=api_key)

    # Format context for the prompt
    context_str = "\n\n".join(
        f"[Source {i+1}]: {res['content']}" for i, res in enumerate(context)
    )

    messages = [
        {
            "role": "system",
            "content": "You are a knowledgeable assistant. Use the provided context to answer questions accurately and concisely.",
        },
        {"role": "user", "content": f"Context:\n{context_str}\n\nQuestion: {question}"},
    ]

    try:
        response = client.chat.completions.create(
            model=GPT_MODEL, messages=messages, temperature=0.3, max_tokens=1024
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Answer generation failed: {str(e)}")
        return "Error generating answer"


def generate_media(media_type, text, session_id, api_key=None):
    """
    Generate different media types from text using Replicate API or OpenAI:
    - 'poster': Uses ideogram/ideogram-v3-turbo (Replicate)
    - 'video': Uses bytedance/seedance-1-lite (Replicate)
    - 'slides': PPTX presentation (using OpenAI GPT)
    - 'memo': DOCX memo (using OpenAI GPT and python-docx)
    Returns the absolute file path. Raises on any failure.
    """
    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    safe_session = session_id or "anon"
    filename_base = f"{media_type}_{safe_session}_{timestamp}"
    out_path = os.path.join(OUTPUT_FOLDER, f"{filename_base}.{media_type}")

    # Replicate API token required for poster and video
    if media_type in ["poster", "video"] and not REPLICATE_API_TOKEN:
        raise RuntimeError("REPLICATE_API_TOKEN environment variable not set")

    # OpenAI API key required for slides and memo
    if media_type in ["slides", "memo"] and not api_key:
        raise RuntimeError("OpenAI API key is required for slides and memo generation")

    # Generate poster using Ideogram
    if media_type == "poster":
        out_path = os.path.join(OUTPUT_FOLDER, f"{filename_base}.png")
        try:
            output = replicate.run(
                "ideogram/ideogram-v3-turbo:7e6f1d5c2f3e3f9bccf1d3e6e1e1f6c1a7a8b5c3d0a5a5a5a5a5a5a5a5a5a5a5a5",
                input={
                    "prompt": text[:500],  # Truncate to 500 characters
                    "width": 1024,
                    "height": 1024,
                    "num_outputs": 1,
                    "guidance_scale": 7.5,
                },
            )

            # Download the generated image
            image_url = output[0]
            response = requests.get(image_url)
            response.raise_for_status()
            with open(out_path, "wb") as f:
                f.write(response.content)

            return os.path.abspath(out_path)

        except Exception as e:
            logger.error(f"Poster generation error: {e}")
            raise RuntimeError(f"Poster generation failed: {str(e)}")

    # Generate video using Seedance
    elif media_type == "video":
        out_path = os.path.join(OUTPUT_FOLDER, f"{filename_base}.mp4")
        try:
            output = replicate.run(
                "bytedance/seedance-1-lite:5d4d1a9f6c8b2d7c3d1d3e3f3d3f3d3f3d3f3d3f3d3f3d3f",
                input={
                    "prompt": text[:500],  # Truncate to 500 characters
                    "video_length": "5s",  # 5 seconds video
                    "fps": 24,
                    "seed": 42,
                },
            )

            # Download the generated video
            video_url = output["video"]
            response = requests.get(video_url)
            response.raise_for_status()
            with open(out_path, "wb") as f:
                f.write(response.content)

            return os.path.abspath(out_path)

        except Exception as e:
            logger.error(f"Video generation error: {e}")
            raise RuntimeError(f"Video generation failed: {str(e)}")

    # Generate slides locally using OpenAI
    elif media_type == "slides":
        out_path = os.path.join(OUTPUT_FOLDER, f"{filename_base}.pptx")
        try:
            client = OpenAI(api_key=api_key)
            system_msg = {
                "role": "system",
                "content": "Convert this content into 3-5 slides in JSON format: "
                "[{'title':..., 'bullets':[...]}]",
            }
            resp = client.chat.completions.create(
                model="gpt-4",
                messages=[system_msg, {"role": "user", "content": text}],
                temperature=0.2,
            )
            slides = json.loads(resp.choices[0].message.content)
            prs = Presentation()
            for slide_data in slides:
                slide = prs.slides.add_slide(prs.slide_layouts[1])
                slide.shapes.title.text = slide_data.get("title", "")
                tf = slide.shapes.placeholders[1].text_frame
                for bullet in slide_data.get("bullets", []):
                    tf.add_paragraph().text = bullet
            prs.save(out_path)
            return os.path.abspath(out_path)
        except Exception as e:
            logger.error(f"Slide generation error: {e}")
            raise RuntimeError(f"Slide generation failed: {str(e)}")

    # Generate memo locally using OpenAI
    elif media_type == "memo":
        out_path = os.path.join(OUTPUT_FOLDER, f"{filename_base}.docx")
        try:
            client = OpenAI(api_key=api_key)
            resp = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "user",
                        "content": f"Format this as a professional memo:\n\n{text}",
                    }
                ],
                temperature=0.3,
            )
            memo_content = resp.choices[0].message.content
            doc = DocxWriter()
            doc.add_heading("Memo", level=1)
            for para in memo_content.splitlines():
                if para.strip():
                    doc.add_paragraph(para.strip())
            doc.save(out_path)
            return os.path.abspath(out_path)
        except Exception as e:
            logger.error(f"Memo generation error: {e}")
            raise RuntimeError(f"Memo generation failed: {str(e)}")

    else:
        raise ValueError(f"Unknown media type: {media_type!r}")


def main():
    """Main RAG pipeline execution"""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY environment variable not set")
        return

    # Get question from environment or use default
    question = os.getenv("QUESTION", "What is the main topic of these documents?")

    # Check for existing index or create new one
    index_path = Path(OUTPUT_FOLDER) / "index.faiss"
    metadata_path = Path(OUTPUT_FOLDER) / "metadata.json"

    if index_path.exists() and metadata_path.exists():
        logger.info("Loading existing index")
        index = faiss.read_index(str(index_path))
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
    else:
        logger.info("Building new index")
        index, metadata = create_index(api_key)
        faiss.write_index(index, str(index_path))
        with open(metadata_path, "w") as f:
            json.dump(metadata, f)

    # Process the question
    logger.info("Processing question")
    question_embedding = get_embeddings([question], api_key)[0]
    context = search_index(index, metadata, question_embedding)
    answer = generate_answer(question, context, api_key)

    # Save results
    result = {
        "question": question,
        "answer": answer,
        "sources": [
            {"file": item["filename"], "chunk": item["chunk_index"]} for item in context
        ],
        "timestamp": datetime.now().isoformat(),
    }

    with open(Path(OUTPUT_FOLDER) / "result.json", "w") as f:
        json.dump(result, f, indent=2)

    logger.info(f"Answer: {answer}")
    logger.info("RAG pipeline completed")


if __name__ == "__main__":
    main()
