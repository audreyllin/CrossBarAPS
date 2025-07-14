import os
import json
import base64
import logging
import re
import time
import subprocess
import numpy as np
from datetime import datetime
from pathlib import Path

# Document processing imports
import pytesseract
import pdfplumber
import fitz
from PIL import Image, ImageDraw, ImageFont
from docx import Document as DocxReader
from docx import Document as DocxWriter
from pptx import Presentation
from pptx.util import Inches
from openai import OpenAI

# AI/Vector imports
import faiss
import requests
from pptx.util import Pt
from gtts import gTTS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("optimized_rag")

# Configuration
UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "output"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
LAST_QUESTION_PATH = Path(OUTPUT_FOLDER) / "last_question.txt"

CHUNK_SIZE = 1500
CHUNK_OVERLAP = 300
EMBEDDING_MODEL = "text-embedding-3-large"
GPT_MODEL = "gpt-4-turbo"
MIN_CHUNK_LENGTH = 100
MAX_FILE_SIZE_MB = 10

# Configuration for pre-signed templates
TEMPLATES_DIR = "templates"
os.makedirs(TEMPLATES_DIR, exist_ok=True)

# Pre-signed templates configuration
TEMPLATES = {
    "poster": {
        "template_id": "canva_12345",
        "placeholders": ["title", "subtitle", "content"],
    },
    "gamma_slides": {
        "template_id": "gamma_67890",
        "placeholders": ["slide1_title", "slide1_content", "slide2_title"],
    },
    "slidesgpt": {
        "template_id": "slidesgpt_abcde",
        "placeholders": ["title", "points", "conclusion"],
    },
}

# Video service configuration
VIDEO_SERVICES = {
    "runwayml": {
        "api_url": "https://api.runwayml.com/video/generate",
        "env_key": "RUNWAYML_API_KEY",
    },
    "pika": {"api_url": "https://api.pika.art/v1/generate", "env_key": "PIKA_API_KEY"},
    "kaiber": {
        "api_url": "https://api.kaiber.ai/v1/generate",
        "env_key": "KAIBER_API_KEY",
    },
}


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
                index.add(np.array([emb]))
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
    distances, indices = index.search(np.array([query_embedding]), k)
    return [metadata[i] for i in indices[0] if i < len(metadata)]


def generate_answer(question, context, api_key, last_question=None):
    """Generate a focused, technical answer using GPT based on context and question."""
    client = OpenAI(api_key=api_key)

    # Check if it's a vague follow-up (e.g., "Can you elaborate?")
    vague_followups = {
        "can you elaborate?",
        "elaborate",
        "what does that mean?",
        "explain more",
        "tell me more",
    }
    is_followup = question.strip().lower() in vague_followups

    # Determine adjusted question if needed
    adjusted_question = question
    if is_followup and last_question:
        adjusted_question = (
            f"Please elaborate in more technical detail about the previous topic: '{last_question}'. "
            "Focus on its technical foundations and implementation, including specific examples if available."
        )

    # Format context
    context_str = "\n\n".join(
        f"[Source {i+1}]: {res['content']}" for i, res in enumerate(context)
    )

    # Adjust system prompt
    system_prompt = (
        "You are a highly knowledgeable technical assistant specializing in crypto wallets, cryptographic SDKs, "
        "blockchain integrations, and enterprise infrastructure. Your responses must be technically precise, "
        "well-written, and based strictly on the provided context.\n\n"
        "When answering:\n"
        "- For direct questions, give clear and complete answers using real examples (e.g., which blockchains are supported).\n"
        "- For follow-up questions like 'Can you elaborate?', continue from the previous technical point—not a general overview.\n"
        "- Prioritize layered technical depth with strong prose. Avoid repetition and bullet points.\n\n"
        "Mention specific Layer 1, Layer 2, or emerging blockchain ecosystems if relevant to the topic."
    )

    # Build messages
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": f"Context:\n{context_str}\n\nQuestion: {adjusted_question}",
        },
    ]

    try:
        response = client.chat.completions.create(
            model=GPT_MODEL,
            messages=messages,
            temperature=0.3,
            max_tokens=1024,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Answer generation failed: {str(e)}")
        return "Error generating answer"


# Media generation functions
def generate_media(media_type, answer, session_id, api_key):
    """Generate media based on type"""
    if media_type == "video":
        return generate_video(answer, session_id, api_key)
    elif media_type == "poster":
        return generate_canva_poster(answer, session_id)
    elif media_type == "slides":
        return generate_slidesgpt_presentation(answer, session_id, api_key)
    elif media_type == "memo":
        return generate_openai_memo(answer, session_id, api_key)
    else:
        raise ValueError(f"Unsupported media type: {media_type}")


def generate_video(answer, session_id, api_key):
    """Generate video using fallback services with priority"""
    try:
        # 1. Check for pre-signed template first
        template_path = os.path.join(TEMPLATES_DIR, "video_template.mp4")
        if os.path.exists(template_path):
            logger.info("Using pre-signed video template")
            return template_path

        # 2. Try PixVerse with timeout handling
        try:
            logger.info("Attempting PixVerse API")
            return generate_pixverse_video(answer, session_id, api_key)
        except Exception as e:
            logger.warning(f"PixVerse failed: {str(e)}")

        # 3. Try alternative services
        for service, config in VIDEO_SERVICES.items():
            service_key = os.getenv(config["env_key"])
            if service_key:
                try:
                    logger.info(f"Trying {service} API")
                    if service == "runwayml":
                        return generate_runwayml_video(answer, session_id, service_key)
                    elif service == "pika":
                        return generate_pika_video(answer, session_id, service_key)
                    elif service == "kaiber":
                        return generate_kaiber_video(answer, session_id, service_key)
                except Exception as e:
                    logger.warning(f"{service} failed: {str(e)}")

        # 4. Fallback to local ffmpeg with TTS
        logger.info("Using local ffmpeg fallback")
        return generate_local_video(answer, session_id)

    except Exception as e:
        logger.error(f"All video generation failed: {str(e)}")
        raise RuntimeError(f"Video generation failed: {str(e)}")


def generate_pixverse_video(answer, session_id, api_key):
    """Generate video using PixVerse API"""
    try:
        payload = {
            "prompt": answer,
            "style": "realistic",
            "api_key": api_key,
        }

        # Initiate video generation
        generate_url = "https://api.pixverse.ai/generate"
        response = requests.post(generate_url, json=payload, timeout=30)
        response.raise_for_status()

        # Check status and get video URL
        task_id = response.json().get("task_id")
        if not task_id:
            raise RuntimeError("No task ID returned from PixVerse API")

        status_url = f"https://api.pixverse.ai/tasks/{task_id}"
        video_url = None

        # Poll for completion
        for _ in range(10):
            status_resp = requests.get(status_url, timeout=10)
            status_resp.raise_for_status()
            status_data = status_resp.json()

            if status_data.get("status") == "completed":
                video_url = status_data.get("video_url")
                break

            time.sleep(15)

        if not video_url:
            raise RuntimeError("Video generation timed out")

        # Download generated video
        output_path = os.path.join(OUTPUT_FOLDER, f"video_{session_id}.mp4")
        download_file(video_url, output_path)
        return output_path

    except requests.exceptions.ConnectionError:
        raise RuntimeError("PixVerse host not found. Check network configuration.")
    except Exception as e:
        raise RuntimeError(f"PixVerse API error: {str(e)}")


def generate_runwayml_video(answer, session_id, api_key):
    """Generate video using RunwayML API"""
    payload = {"prompt": answer, "length": 5, "api_key": api_key}  # seconds

    response = requests.post(
        VIDEO_SERVICES["runwayml"]["api_url"], json=payload, timeout=30
    )
    response.raise_for_status()

    result = response.json()
    video_url = result.get("video_url")
    if not video_url:
        raise RuntimeError("No video URL in RunwayML response")

    output_path = os.path.join(OUTPUT_FOLDER, f"runway_video_{session_id}.mp4")
    download_file(video_url, output_path)
    return output_path


def generate_pika_video(answer, session_id, api_key):
    """Generate video using Pika Labs API"""
    payload = {"text_prompt": answer, "api_key": api_key}

    response = requests.post(
        VIDEO_SERVICES["pika"]["api_url"], json=payload, timeout=30
    )
    response.raise_for_status()

    result = response.json()
    video_url = result.get("video_url")
    if not video_url:
        raise RuntimeError("No video URL in Pika response")

    output_path = os.path.join(OUTPUT_FOLDER, f"pika_video_{session_id}.mp4")
    download_file(video_url, output_path)
    return output_path


def generate_kaiber_video(answer, session_id, api_key):
    """Generate video using Kaiber API"""
    payload = {"prompt": answer, "api_key": api_key}

    response = requests.post(
        VIDEO_SERVICES["kaiber"]["api_url"], json=payload, timeout=30
    )
    response.raise_for_status()

    result = response.json()
    video_url = result.get("video_url")
    if not video_url:
        raise RuntimeError("No video URL in Kaiber response")

    output_path = os.path.join(OUTPUT_FOLDER, f"kaiber_video_{session_id}.mp4")
    download_file(video_url, output_path)
    return output_path


def generate_local_video(answer, session_id):
    """Generate video locally using ffmpeg with TTS and image montage"""
    try:
        # Step 1: Generate TTS audio
        tts = gTTS(text=answer, lang="en")
        audio_path = os.path.join(OUTPUT_FOLDER, f"tts_{session_id}.mp3")
        tts.save(audio_path)

        # Step 2: Create image montage
        image_path = os.path.join(OUTPUT_FOLDER, f"montage_{session_id}.jpg")
        img = Image.new("RGB", (800, 600), color=(73, 109, 137))
        d = ImageDraw.Draw(img)
        d.text((100, 300), "AI Generated Content", fill=(255, 255, 0))
        img.save(image_path)

        # Step 3: Combine with ffmpeg
        video_path = os.path.join(OUTPUT_FOLDER, f"local_video_{session_id}.mp4")
        cmd = [
            "ffmpeg",
            "-loop",
            "1",
            "-i",
            image_path,
            "-i",
            audio_path,
            "-c:v",
            "libx264",
            "-tune",
            "stillimage",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-pix_fmt",
            "yuv420p",
            "-shortest",
            video_path,
        ]
        subprocess.run(cmd, check=True)

        return video_path
    except Exception as e:
        raise RuntimeError(f"Local video generation failed: {str(e)}")


def generate_canva_poster(answer, session_id):
    """Generate poster using Canva API with pre-signed template"""
    try:
        # Auto-fill template with answer content
        content_parts = answer.split("\n")
        fill_data = {
            "title": content_parts[0] if len(content_parts) > 0 else "Key Insights",
            "subtitle": (
                content_parts[1] if len(content_parts) > 1 else "Generated Report"
            ),
            "content": (
                "\n".join(content_parts[2:]) if len(content_parts) > 2 else answer
            ),
        }

        # Generate image
        output_path = os.path.join(OUTPUT_FOLDER, f"poster_{session_id}.png")
        create_poster_image(fill_data, output_path)
        return output_path
    except Exception as e:
        logger.error(f"Poster generation error: {e}")
        raise RuntimeError(f"Poster generation failed: {str(e)}")


def generate_slidesgpt_presentation(answer, session_id, api_key):
    """Generate presentation using SlidesGPT-style formatting"""
    try:
        # Format content for presentation
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=GPT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "Format this content into a slide presentation structure with titles and bullet points.",
                },
                {"role": "user", "content": answer},
            ],
            temperature=0.3,
        )
        structured_content = response.choices[0].message.content.strip()

        # Create presentation
        output_path = os.path.join(OUTPUT_FOLDER, f"slides_{session_id}.pptx")
        prs = Presentation()

        # Title slide
        slide = prs.slides.add_slide(prs.slide_layouts[0])
        title = slide.shapes.title
        title.text = "AI Generated Presentation"

        # Content slides
        for i, section in enumerate(structured_content.split("\n\n")):
            if not section.strip():
                continue

            slide = prs.slides.add_slide(prs.slide_layouts[1])
            title = slide.shapes.title
            content = slide.placeholders[1]

            lines = section.split("\n")
            title.text = lines[0] if lines else f"Slide {i+1}"

            if len(lines) > 1:
                tf = content.text_frame
                for line in lines[1:]:
                    p = tf.add_paragraph()
                    p.text = line
                    p.level = 0

        prs.save(output_path)
        return output_path

    except Exception as e:
        logger.error(f"Presentation generation error: {e}")
        raise RuntimeError(f"Presentation generation failed: {str(e)}")


def generate_openai_memo(answer, session_id, api_key):
    """Generate memo using pre-signed template"""
    try:
        # Create memo
        output_path = os.path.join(OUTPUT_FOLDER, f"memo_{session_id}.docx")
        doc = DocxWriter()

        # Add formatted content
        doc.add_heading("Company Memo", level=0)
        doc.add_paragraph(datetime.now().strftime("%B %d, %Y"))
        doc.add_paragraph("")

        # Split answer into paragraphs
        for paragraph in answer.split("\n\n"):
            if paragraph.strip():
                p = doc.add_paragraph(paragraph)
                p.style.font.size = Pt(11)

        # Add footer
        footer = doc.sections[0].footer
        footer_para = footer.paragraphs[0]
        footer_para.text = "Confidential - Generated by Crossbar AI"

        doc.save(output_path)
        return output_path
    except Exception as e:
        logger.error(f"Memo generation error: {e}")
        raise RuntimeError(f"Memo generation failed: {str(e)}")


# Helper functions
def download_file(url, output_path):
    """Download file from URL with progress tracking"""
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))
    downloaded = 0
    chunk_size = 8192

    with open(output_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    progress = downloaded / total_size * 100
                    logger.info(f"Download progress: {progress:.1f}%")

    return output_path


def create_poster_image(fill_data, output_path):
    """Create poster image for demo purposes"""
    # Create a blank image
    width, height = 1200, 1800
    img = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    try:
        font_title = ImageFont.truetype("arialbd.ttf", 72)
    except:
        font_title = ImageFont.load_default()

    try:
        font_subtitle = ImageFont.truetype("arial.ttf", 48)
    except:
        font_subtitle = ImageFont.load_default()

    try:
        font_content = ImageFont.truetype("arial.ttf", 36)
    except:
        font_content = ImageFont.load_default()

    # Draw title
    draw.text((100, 200), fill_data["title"], font=font_title, fill=(0, 0, 0))

    # Draw subtitle
    draw.text(
        (100, 300), fill_data["subtitle"], font=font_subtitle, fill=(100, 100, 100)
    )

    # Draw content
    y_position = 400
    for line in fill_data["content"].split("\n"):
        draw.text((100, y_position), line, font=font_content, fill=(0, 0, 0))
        y_position += 50

    # Add footer
    draw.text(
        (100, height - 100),
        "Generated by Crossbar AI",
        font=font_subtitle,
        fill=(150, 150, 150),
    )

    img.save(output_path)


def main():
    """Main RAG pipeline execution"""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY environment variable not set")
        return

    # Get question from environment or use default
    question = os.getenv("QUESTION", "What is the main topic of these documents?")
    
    # Reset handling
    if question.strip().lower() == "reset":
        LAST_QUESTION_PATH.unlink(missing_ok=True)
        logger.info("Last question reset.")
        return

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

    # Load last question from persistent file if available
    prior_question = None
    if LAST_QUESTION_PATH.exists():
        with open(LAST_QUESTION_PATH, "r", encoding="utf-8") as f:
            prior_question = f.read().strip()

    # Generate the answer with follow-up awareness
    answer = generate_answer(question, context, api_key, last_question=prior_question)

    # Save current question as new last_question
    with open(LAST_QUESTION_PATH, "w", encoding="utf-8") as f:
        f.write(question.strip())

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