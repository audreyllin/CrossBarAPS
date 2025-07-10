import os
import logging
import pytesseract
import pdfplumber
from PIL import Image
from docx import Document as DocxDoc
from pptx import Presentation
from openai import OpenAI
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Text extraction from files using GPT when possible
def extract_text_from_file(filepath):
    """Extract text from various file formats"""
    path = Path(filepath)
    ext = path.suffix.lower()
    text = ""

    try:
        if ext == ".pdf":
            # For PDFs, try to extract text normally
            with pdfplumber.open(filepath) as pdf:
                text = "\n".join(
                    page.extract_text() for page in pdf.pages if page.extract_text()
                )

            # If text extraction fails, use OCR with GPT vision
            if not text.strip():
                return extract_with_gpt_vision(filepath, "PDF")

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

        elif ext in [".jpg", ".jpeg", ".png"]:
            # Use GPT Vision API for image text extraction
            return extract_with_gpt_vision(filepath, "image")

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


def extract_with_gpt_vision(filepath, file_type):
    """Extract text using GPT Vision API"""
    try:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set")

        client = OpenAI(api_key=api_key)

        # Create prompt based on file type
        if file_type == "PDF":
            prompt = "Extract all text from this PDF document. Include all headings, paragraphs, and bullet points."
        elif file_type == "image":
            prompt = "Extract all text from this image. Include any visible text, numbers, and symbols."
        else:
            prompt = "Extract all text from this document."

        with open(filepath, "rb") as image_file:
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
                                    "url": f"data:image/jpeg;base64,{image_file.read().hex()}"
                                },
                            },
                        ],
                    }
                ],
                max_tokens=4096,
            )

        return response.choices[0].message.content

    except Exception as e:
        logger.error(f"GPT Vision extraction failed: {str(e)}")
        # Fallback to OCR for images
        if file_type == "image":
            try:
                image = Image.open(filepath)
                return pytesseract.image_to_string(image)
            except:
                return "Error extracting text"
        return "Error extracting text"


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
