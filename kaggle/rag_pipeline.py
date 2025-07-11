import os
import json
import time
import logging
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
import numpy as np
import faiss
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rag_pipeline")

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "output"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Text chunking parameters
CHUNK_SIZE = 1000  # characters per chunk
CHUNK_OVERLAP = 200  # overlapping characters between chunks
EMBEDDING_MODEL = "text-embedding-3-large"
GPT_MODEL = "gpt-4"
MIN_CHUNK_LENGTH = 50  # minimum chunk length to consider


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split text into overlapping chunks"""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks


def clean_text(text):
    """Clean and normalize text"""
    # Remove excessive whitespace
    text = re.sub(r"\s+", " ", text)
    # Remove non-printable characters
    text = "".join(char for char in text if char.isprintable())
    return text.strip()


def is_image_heavy_pdf(filepath):
    try:
        doc = fitz.open(filepath)
        return sum(1 for p in doc if len(p.get_images(full=True)) > 0) / len(doc) > 0.5
    except:
        return False


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
                logger.info(f"Using OCR for PDF: {filepath}")
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
        logger.error(f"[extract_text] {filepath} failed: {e}")

    return clean_text(text)[:50000]  # Limit to 50k characters


def get_embeddings(texts, api_key):
    """Get embeddings for multiple text chunks"""
    client = OpenAI(api_key=api_key)
    response = client.embeddings.create(
        model=EMBEDDING_MODEL, input=texts, encoding_format="float"
    )
    return [data.embedding for data in response.data]


def create_vector_index(api_key):
    """Create FAISS vector index from all files in uploads folder"""
    index = faiss.IndexFlatL2(3072)  # Dimension for text-embedding-3-large
    metadata = []

    for filepath in Path(UPLOAD_FOLDER).glob("*"):
        if filepath.is_file():
            try:
                logger.info(f"Processing for vector index: {filepath.name}")
                text = extract_text_from_file(filepath)

                if not text.strip():
                    logger.warning(f"Skipping empty file: {filepath.name}")
                    continue

                # Split text into chunks
                chunks = chunk_text(text)
                embeddings = get_embeddings(chunks, api_key)

                # Add to index
                for i, embedding in enumerate(embeddings):
                    # Only add meaningful chunks
                    if len(chunks[i]) > MIN_CHUNK_LENGTH:
                        index.add(np.array([embedding]))
                        metadata.append(
                            {
                                "filename": filepath.name,
                                "chunk_index": i,
                                "content": chunks[i][:500],  # Store preview
                            }
                        )

                time.sleep(0.1)  # Avoid rate limits
            except Exception as e:
                logger.error(f"Error processing {filepath.name}: {str(e)}")

    return index, metadata


def search_index(index, metadata, query_embedding, k=5):
    """Search index for top k results"""
    distances, indices = index.search(np.array([query_embedding]), k)
    results = []
    for i in indices[0]:
        if i >= 0 and i < len(metadata):
            results.append(metadata[i])
    return results


def generate_answer(question, context, api_key):
    """Generate answer using GPT-4 with context"""
    client = OpenAI(api_key=api_key)

    # Build prompt with context
    context_str = "\n\n".join(
        [f"Source {i+1}: {res['content']}..." for i, res in enumerate(context)]
    )
    prompt = f"""
    Context information:
    {context_str}
    
    Question: {question}
    Answer:
    """

    response = client.chat.completions.create(
        model=GPT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1024,
        temperature=0.3,
    )
    return response.choices[0].message.content.strip()


def process_uploads(api_key):
    """Process all files in uploads folder and create vector index"""
    logger.info("Starting uploads processing")
    return create_vector_index(api_key)


if __name__ == "__main__":
    # Get API key from environment
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY environment variable not set")
        exit(1)

    # Step 1: Process uploads and create vector index
    index, metadata = process_uploads(api_key)

    # Step 2: Get question from environment
    question = os.getenv("QUESTION", "What is our company's mission?")
    logger.info(f"Processing question: {question}")

    # Step 3: Embed question
    question_embedding = get_embeddings([question], api_key)[0]

    # Step 4: Search for relevant context
    context = search_index(index, metadata, question_embedding, k=5)

    # Step 5: Generate answer
    answer = generate_answer(question, context, api_key)

    # Step 6: Save results
    result = {
        "question": question,
        "answer": answer,
        "context_sources": [
            {
                "filename": res["filename"],
                "chunk_index": res["chunk_index"],
                "content_preview": res["content"],
            }
            for res in context
        ],
        "generated_at": datetime.now().isoformat(),
    }

    with open(Path(OUTPUT_FOLDER) / "result.json", "w") as f:
        json.dump(result, f, indent=2)

    logger.info("RAG pipeline completed. Results saved to result.json")
