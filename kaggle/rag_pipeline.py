import os
import json
import logging
from pathlib import Path
from datetime import datetime
import pytesseract
import pdfplumber
import fitz
from PIL import Image
from openai import OpenAI
import numpy as np
import faiss
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("optimized_rag")

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "output"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Configuration
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 300
EMBEDDING_MODEL = "text-embedding-3-large"
GPT_MODEL = "gpt-4-turbo"
MIN_CHUNK_LENGTH = 100
MAX_FILE_SIZE_MB = 10


def clean_text(text):
    """Optimized text cleaning"""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def is_image_heavy_pdf(filepath):
    """Efficient PDF image detection"""
    try:
        doc = fitz.open(filepath)
        image_count = sum(len(page.get_images(full=True)) for page in doc)
        return image_count / len(doc) > 0.5
    except:
        return False


def extract_text(filepath):
    """Optimized text extraction"""
    ext = Path(filepath).suffix.lower()
    try:
        # Skip large files
        if os.path.getsize(filepath) > MAX_FILE_SIZE_MB * 1024 * 1024:
            logger.warning(f"Skipping large file: {filepath}")
            return ""

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
            logger.warning(f"Unsupported format: {ext}")
            return ""
    except Exception as e:
        logger.error(f"Extraction error: {str(e)}")
        return ""


def extract_pdf_text(filepath):
    """Optimized PDF text extraction"""
    text = ""
    try:
        # Try text-based extraction first
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
    return clean_text(text)[:50000]


def extract_docx_text(filepath):
    """Efficient DOCX extraction"""
    try:
        from docx import Document

        doc = Document(filepath)
        return clean_text("\n".join(p.text for p in doc.paragraphs))
    except Exception as e:
        logger.error(f"DOCX extraction failed: {str(e)}")
        return ""


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Efficient text chunking with sentence boundary awareness"""
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))

        # Adjust to nearest sentence boundary
        if end < len(text):
            boundary = max(
                text.rfind(".", start, end),
                text.rfind("?", start, end),
                text.rfind("!", start, end),
            )
            if boundary > start and (boundary - start) > MIN_CHUNK_LENGTH:
                end = boundary + 1

        chunks.append(text[start:end])
        start = end - overlap

    return chunks


def get_embeddings(texts, api_key):
    """Batch embedding generation with error handling"""
    client = OpenAI(api_key=api_key)
    try:
        response = client.embeddings.create(
            model=EMBEDDING_MODEL, input=texts, encoding_format="float"
        )
        return [data.embedding for data in response.data]
    except Exception as e:
        logger.error(f"Embedding failed: {str(e)}")
        return []


def create_index(api_key):
    """Efficient indexing with metadata tracking"""
    index = faiss.IndexFlatL2(3072)
    metadata = []
    valid_extensions = [".pdf", ".doc", ".docx", ".txt", ".pptx", ".xlsx"]

    for filepath in Path(UPLOAD_FOLDER).glob("*"):
        if filepath.suffix.lower() not in valid_extensions:
            continue

        try:
            logger.info(f"Processing: {filepath.name}")
            text = extract_text(filepath)
            if not text:
                continue

            chunks = chunk_text(text)
            if not chunks:
                continue

            # Batch processing for efficiency
            batch_size = 10
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i : i + batch_size]
                embeddings = get_embeddings(batch, api_key)

                for j, emb in enumerate(embeddings):
                    idx = i + j
                    if idx < len(chunks) and len(chunks[idx]) > MIN_CHUNK_LENGTH:
                        index.add(np.array([emb]))
                        metadata.append(
                            {
                                "filename": filepath.name,
                                "chunk_index": idx,
                                "content": chunks[idx][:500],
                            }
                        )
        except Exception as e:
            logger.error(f"Indexing error: {str(e)}")

    return index, metadata


def search_index(index, metadata, query_embedding, k=5):
    """Efficient FAISS search"""
    distances, indices = index.search(np.array([query_embedding]), k)
    return [metadata[i] for i in indices[0] if i < len(metadata)]


def generate_answer(question, context, api_key):
    """Optimized answer generation"""
    client = OpenAI(api_key=api_key)

    # Build context string
    context_str = "\n\n".join(
        f"[Source {i+1}]: {res['content']}" for i, res in enumerate(context)
    )

    # System message for better control
    system_msg = {
        "role": "system",
        "content": "You are a knowledgeable assistant. Use the provided context to answer questions accurately and concisely.",
    }

    user_msg = {
        "role": "user",
        "content": f"Context:\n{context_str}\n\nQuestion: {question}",
    }

    try:
        response = client.chat.completions.create(
            model=GPT_MODEL,
            messages=[system_msg, user_msg],
            max_tokens=1024,
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Generation failed: {str(e)}")
        return "Error generating answer"


def main():
    # Get API key from environment
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY environment variable not set")
        return

    question = os.getenv("QUESTION", "What is our company's mission?")

    # Create or load index
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

    # Embed question
    logger.info("Embedding question")
    q_embedding = get_embeddings([question], api_key)[0]

    # Retrieve context
    logger.info("Searching for relevant context")
    context = search_index(index, metadata, q_embedding, k=5)

    # Generate answer
    logger.info("Generating answer")
    answer = generate_answer(question, context, api_key)

    # Save results
    result = {
        "question": question,
        "answer": answer,
        "context_sources": [
            {"filename": res["filename"], "chunk_index": res["chunk_index"]}
            for res in context
        ],
        "generated_at": datetime.now().isoformat(),
    }

    with open(Path(OUTPUT_FOLDER) / "result.json", "w") as f:
        json.dump(result, f, indent=2)
    logger.info("RAG pipeline completed")


if __name__ == "__main__":
    main()
