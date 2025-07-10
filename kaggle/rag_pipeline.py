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
from pdf2image import convert_from_bytes

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Text extraction from files using GPT when possible
def extract_text_from_file(filepath):
    """Extract text from various file formats with visual analysis"""
    path = Path(filepath)
    ext = path.suffix.lower()
    text = ""

    try:
        if ext == ".pdf":
            # First try text extraction
            with pdfplumber.open(filepath) as pdf:
                text = "\n".join(
                    page.extract_text() for page in pdf.pages if page.extract_text()
                )

            # If text extraction fails or sparse, use GPT-Vision
            if not text.strip() or is_image_heavy_pdf(filepath):
                return extract_with_gpt_vision_base64(filepath)

        elif ext in [".jpg", ".jpeg", ".png"]:
            # Use GPT Vision API for image text extraction
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
        logger.error(f"Text extraction failed: {str(e)}")
        return f"Error extracting text: {str(e)}"

    return text[:20000]  # Limit to 20k characters


def extract_with_gpt_vision_base64(filepath):
    """Extract text using GPT Vision API with base64 encoding"""
    try:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set")

        client = OpenAI(api_key=api_key)

        # Read file as base64
        with open(filepath, "rb") as file:
            base64_image = base64.b64encode(file.read()).decode("utf-8")

        # Create prompt
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
        # Fallback to OCR
        return extract_with_ocr(filepath)


def extract_image_metadata(filepath):
    """Get detailed image analysis with GPT Vision"""
    try:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set")

        client = OpenAI(api_key=api_key)

        # Read file as base64
        with open(filepath, "rb") as file:
            base64_image = base64.b64encode(file.read()).decode("utf-8")

        # Create detailed analysis prompt
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
    """Fallback to OCR when GPT Vision fails"""
    try:
        # For PDFs, convert to images first
        if filepath.lower().endswith(".pdf"):
            images = convert_from_bytes(open(filepath, "rb").read())
            text = ""
            for img in images:
                text += pytesseract.image_to_string(img) + "\n"
            return text

        # For images, use direct OCR
        return pytesseract.image_to_string(Image.open(filepath))

    except Exception as e:
        logger.error(f"OCR extraction failed: {str(e)}")
        return "Text extraction failed"


def is_image_heavy_pdf(filepath, threshold=0.5):
    """Check if PDF is image-heavy"""
    try:
        with pdfplumber.open(filepath) as pdf:
            total_pages = len(pdf.pages)
            image_pages = sum(1 for page in pdf.pages if page.images)
            return (image_pages / total_pages) > threshold if total_pages > 0 else False
    except:
        return False


def embed_and_store_file(filepath, api_key, model="text-embedding-3-large"):
    """Process a file: extract text, embed with GPT, and store"""
    try:
        # 1. Extract text
        text_content = extract_text_from_file(filepath)
        if not text_content.strip():
            raise ValueError(f"No extractable content in file: {filepath}")

        # 2. Generate embedding with GPT
        client = OpenAI(api_key=api_key)
        response = client.embeddings.create(
            model=model,
            input=[text_content[:8191]],  # Stay within token limits
            encoding_format="float",
        )
        vector = np.array(response.data[0].embedding)

        # 3. Return results for storage
        return {
            "text": text_content,
            "vector": vector,
            "filename": os.path.basename(filepath),
        }

    except Exception as e:
        logger.error(f"File processing failed: {str(e)}")
        raise


# GPT-based chunk embedding (if needed for large files)
def embed_chunks_with_gpt(chunks, api_key, model="text-embedding-3-large"):
    """Embed text chunks using GPT API"""
    client = OpenAI(api_key=api_key)
    response = client.embeddings.create(
        model=model, input=chunks, encoding_format="float"
    )
    return [np.array(item.embedding) for item in response.data]


if __name__ == "__main__":
    # Test file processing
    test_file = "uploads/sample.pdf"
    if os.path.exists(test_file):
        api_key = os.environ.get("OPENAI_API_KEY")
        if api_key:
            result = embed_and_store_file(test_file, api_key)
            print(f"Processed file: {result['filename']}")
            print(f"Text length: {len(result['text'])} characters")
        else:
            print("OPENAI_API_KEY not set")
