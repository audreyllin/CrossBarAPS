import os
import logging
import pytesseract
import pdfplumber
from PIL import Image
from docx import Document as DocxDoc
from pptx import Presentation
from openai import OpenAI
from pathlib import Path
import numpy as np
import base64
import json
import faiss
from pdf2image import convert_from_bytes

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
UPLOAD_DIR = "uploads"
EMBEDDINGS_DIR = "embeddings"
DB_DIR = "db"
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
os.makedirs(DB_DIR, exist_ok=True)


def extract_text_from_file(filepath):
    """Extract text from various file formats with visual analysis"""
    path = Path(filepath)
    ext = path.suffix.lower()
    text = ""

    try:
        if ext == ".pdf":
            with pdfplumber.open(filepath) as pdf:
                text = "\n".join(
                    page.extract_text() for page in pdf.pages if page.extract_text()
                )
            if not text.strip() or is_image_heavy_pdf(filepath):
                return extract_with_gpt_vision_from_pdf(filepath)

        elif ext in [".jpg", ".jpeg", ".png"]:
            return extract_with_gpt_vision_base64(filepath)

        elif ext in [".doc", ".docx"]:
            doc = DocxDoc(filepath)
            text = "\n".join(para.text for para in doc.paragraphs)

        elif ext in [".ppt", ".pptx"]:
            prs = Presentation(filepath)
            text = "\n".join(
                shape.text
                for slide in prs.slides
                for shape in slide.shapes
                if hasattr(shape, "text") and shape.text
            )

        elif ext == ".txt":
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read()

        else:
            logger.warning(f"Unsupported file type: {ext}")
            return ""

    except Exception as e:
        logger.error(f"Text extraction failed for {filepath}: {str(e)}")
        return f"Error extracting text: {str(e)}"

    return text[:20000]


def extract_with_gpt_vision_from_pdf(filepath):
    try:
        images = convert_from_bytes(open(filepath, "rb").read())
        all_text = ""
        for img in images:
            temp_path = "temp.jpg"
            img.save(temp_path, format="JPEG")
            all_text += extract_with_gpt_vision_base64(temp_path) + "\n"
            os.remove(temp_path)
        return all_text
    except Exception as e:
        logger.error(f"PDF to image conversion failed: {e}")
        return extract_with_ocr(filepath)


def extract_with_gpt_vision_base64(filepath):
    try:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set")

        client = OpenAI(api_key=api_key)
        with open(filepath, "rb") as file:
            base64_image = base64.b64encode(file.read()).decode("utf-8")

        prompt = (
            "Extract all text and describe visual content. "
            "Include headings, paragraphs, captions, and any visible text. "
            "Also describe any diagrams, charts, or significant visual elements."
        )

        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            },
                        },
                    ],
                }
            ],
            max_tokens=2000,
        )

        return response.choices[0].message.content

    except Exception as e:
        logger.error(f"GPT Vision extraction failed: {str(e)}")
        return extract_with_ocr(filepath)


def extract_image_metadata(filepath):
    """Get detailed image analysis with GPT Vision"""
    try:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set")

        client = OpenAI(api_key=api_key)

        with open(filepath, "rb") as file:
            base64_image = base64.b64encode(file.read()).decode("utf-8")

        prompt = (
            "Analyze this image in detail. Describe: "
            "1. All visual elements and their composition "
            "2. Artistic style and techniques used "
            "3. Color palette and lighting "
            "4. Any text or symbols present "
            "5. Overall mood and aesthetic qualities "
            "6. Potential meaning or purpose"
        )

        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            },
                        },
                    ],
                }
            ],
            max_tokens=1500,
        )

        return response.choices[0].message.content

    except Exception as e:
        logger.error(f"Image analysis failed: {str(e)}")
        return "Image analysis not available"

def extract_with_ocr(filepath):
    try:
        suffix = Path(filepath).suffix.lower()
        if suffix == ".pdf":
            images = convert_from_bytes(open(filepath, "rb").read())
            text = ""
            for img in images:
                text += pytesseract.image_to_string(img) + "\n"
            return text
        return pytesseract.image_to_string(Image.open(filepath))
    except Exception as e:
        logger.error(f"OCR extraction failed: {str(e)}")
        return "Text extraction failed"


def is_image_heavy_pdf(filepath, threshold=0.5):
    try:
        with pdfplumber.open(filepath) as pdf:
            total_pages = len(pdf.pages)
            image_pages = sum(1 for page in pdf.pages if page.images)
            return (image_pages / total_pages) > threshold if total_pages > 0 else False
    except:
        return False


def embed_and_store_file(filepath, api_key, model="text-embedding-3-large"):
    try:
        text_content = extract_text_from_file(filepath)
        if not text_content.strip():
            raise ValueError(f"No extractable content in file: {filepath}")

        client = OpenAI(api_key=api_key)
        response = client.embeddings.create(
            model=model,
            input=[text_content[:8191]],
            encoding_format="float",
        )
        vector = np.array(response.data[0].embedding)

        return {
            "text": text_content,
            "vector": vector,
            "filename": os.path.basename(filepath),
        }

    except Exception as e:
        logger.error(f"File processing failed: {str(e)}")
        raise


def embed_chunks_with_gpt(chunks, api_key, model="text-embedding-3-large"):
    client = OpenAI(api_key=api_key)
    response = client.embeddings.create(
        model=model, input=chunks, encoding_format="float"
    )
    return [np.array(item.embedding) for item in response.data]


def chunk_text(text, max_tokens=500):
    paragraphs = text.split("\n")
    chunks, current = [], ""
    for para in paragraphs:
        if len(current) + len(para) < max_tokens:
            current += " " + para
        else:
            chunks.append(current.strip())
            current = para
    if current:
        chunks.append(current.strip())
    return chunks


def save_faiss_index(index, mapping, filename_prefix):
    faiss.write_index(index, f"{DB_DIR}/{filename_prefix}_index.faiss")
    with open(f"{DB_DIR}/{filename_prefix}_mapping.json", "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2)


def process_and_embed_all_documents():
    index = faiss.IndexFlatL2(1536)
    all_chunks = []
    all_text_map = {}
    file_id = 0

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.warning("OPENAI_API_KEY not set. Skipping file auto-processing.")
        return

    for file in Path(UPLOAD_DIR).iterdir():
        if not file.is_file() or file.suffix.lower() not in [".txt", ".pdf", ".docx"]:
            continue

        logger.info(f"\ud83d\udcc4 Processing: {file.name}")
        try:
            result = embed_and_store_file(file, api_key)
            text = result["text"]
            vector = result["vector"]
            filename = result["filename"]

            embedding_path = os.path.join(EMBEDDINGS_DIR, f"{filename}.npy")
            np.save(embedding_path, vector)

            chunks = chunk_text(text)
            embeddings = embed_chunks_with_gpt(chunks, api_key)

            for i, emb in enumerate(embeddings):
                index.add(np.array([emb]))
                all_text_map[len(all_text_map)] = {"file": filename, "chunk": chunks[i]}
            logger.info(f"✅ Embedded {len(chunks)} chunks from {file.name}")

        except Exception as e:
            logger.error(f"[!] Error processing {file.name}: {e}")

    if all_text_map:
        save_faiss_index(index, all_text_map, "combined")
        logger.info("\ud83d\udce6 Saved combined FAISS index and mapping.")
    else:
        logger.info("\u26a0\ufe0f No valid data to embed from uploads.")


if __name__ == "__main__":
    test_file = "uploads/sample.pdf"
    if os.path.exists(test_file):
        api_key = os.environ.get("OPENAI_API_KEY")
        if api_key:
            result = embed_and_store_file(test_file, api_key)
            print(f"Processed file: {result['filename']}")
            print(f"Text length: {len(result['text'])} characters")
        else:
            print("OPENAI_API_KEY not set")
