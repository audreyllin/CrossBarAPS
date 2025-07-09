from flask import Flask, request, jsonify, render_template
import os
import logging
from rag_pipeline import RagProcessor
from openai import (
    OpenAI,
    OpenAIError,
    AuthenticationError,
    RateLimitError,
    BadRequestError,
)
import threading
import uuid
import tiktoken
from werkzeug.utils import secure_filename
import time
import json
import PyPDF2
import pdfplumber  # Replaced pdfminer with pdfplumber
from docx import Document
import hashlib
from datetime import datetime
import re
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
import heapq
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Get absolute path to templates directory
current_dir = os.path.dirname(os.path.abspath(__file__))
template_path = os.path.join(current_dir, "templates")
DB_FILE = os.path.join(current_dir, "memory.db")
VECTOR_STORE = os.path.join(current_dir, "vector_store")
os.makedirs(VECTOR_STORE, exist_ok=True)

app = Flask(__name__, template_folder=template_path)
logging.basicConfig(level=logging.INFO)


# Database Helper Functions
def get_db():
    return sqlite3.connect(DB_FILE)


def init_db():
    with get_db() as db:
        db.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                account_number TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL
            )
        """
        )
        db.execute(
            """
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                summary TEXT NOT NULL,
                sources TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(user_id) REFERENCES users(id)
            )
        """
        )
        db.execute(
            """
            CREATE TABLE IF NOT EXISTS context_vectors (
                session_id TEXT NOT NULL,
                context_id TEXT NOT NULL,
                vector BLOB NOT NULL,
                PRIMARY KEY (session_id, context_id)
            )
        """
        )
        db.commit()


def get_user_by_account(account_number):
    with get_db() as db:
        return db.execute(
            "SELECT * FROM users WHERE account_number = ?", (account_number,)
        ).fetchone()


def create_user(account_number, password):
    password_hash = generate_password_hash(password)
    with get_db() as db:
        db.execute(
            "INSERT INTO users (account_number, password_hash) VALUES (?, ?)",
            (account_number, password_hash),
        )
        db.commit()


def validate_login(account_number, password):
    user = get_user_by_account(account_number)
    return user if user and check_password_hash(user[2], password) else None


def store_memory(user_id, question, answer, summary, sources):
    with get_db() as db:
        db.execute(
            "INSERT INTO memories (user_id, question, answer, summary, sources) VALUES (?, ?, ?, ?, ?)",
            (user_id, question, answer, summary, sources),
        )
        db.commit()


def get_user_memories(user_id):
    with get_db() as db:
        return db.execute(
            "SELECT id, question, answer, summary, sources, created_at FROM memories WHERE user_id = ? ORDER BY created_at DESC",
            (user_id,),
        ).fetchall()


def store_context_vector(session_id, context_id, vector):
    with get_db() as db:
        db.execute(
            "INSERT OR REPLACE INTO context_vectors (session_id, context_id, vector) VALUES (?, ?, ?)",
            (session_id, context_id, json.dumps(vector.tolist())),
        )
        db.commit()


def get_context_vectors(session_id):
    with get_db() as db:
        rows = db.execute(
            "SELECT context_id, vector FROM context_vectors WHERE session_id = ?",
            (session_id,),
        ).fetchall()
    return [(row[0], np.array(json.loads(row[1]))) for row in rows]


# Initialize database
init_db()

# Download NLTK resources
try:
    nltk.download("punkt")
    nltk.download("stopwords")
except:
    logging.warning(
        "NLTK resources not downloaded. Key concept extraction may be limited"
    )

# Initialize RAG processor
try:
    rag_processor = RagProcessor()
    logging.info("RAG processor initialized successfully")
except Exception as e:
    logging.error(f"Failed to initialize RAG processor: {str(e)}")
    rag_processor = None

# Persistent session storage
SESSION_STORE = "sessions"
os.makedirs(SESSION_STORE, exist_ok=True)

# Session management
sessions = {}
sessions_lock = threading.Lock()
UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {
    "txt",
    "pdf",
    "doc",
    "docx",
    "ppt",
    "pptx",
    "jpg",
    "jpeg",
    "png",
    "mp4",
}
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Token counting
encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def count_tokens(text):
    return len(encoding.encode(text))


def trim_conversation_history(history, max_tokens=3000):
    total_tokens = 0
    trimmed_history = []
    if history and history[0]["role"] == "system":
        total_tokens += count_tokens(history[0]["content"])
        trimmed_history.append(history[0])
    for msg in reversed(history[1:]):
        msg_tokens = count_tokens(msg["content"])
        if total_tokens + msg_tokens > max_tokens:
            break
        total_tokens += msg_tokens
        trimmed_history.insert(1, msg)
    return trimmed_history


def get_relevant_contexts(question, session_contexts, session_id, top_n=3):
    if not question or not session_contexts:
        return []

    # Get stored vectors
    stored_vectors = get_context_vectors(session_id)
    if not stored_vectors:
        return []

    # Vectorize question
    vectorizer = TfidfVectorizer()
    try:
        vectorizer.fit([question] + [ctx["content"] for ctx in session_contexts])
    except:
        return []

    question_vec = vectorizer.transform([question]).toarray()[0]

    # Calculate similarities
    similarities = []
    for context_id, ctx_vec in stored_vectors:
        similarity = cosine_similarity([question_vec], [ctx_vec])[0][0]
        similarities.append((context_id, similarity))

    # Get top N contexts
    similarities.sort(key=lambda x: x[1], reverse=True)
    top_contexts = [ctx_id for ctx_id, _ in similarities[:top_n]]

    # Return context objects
    return [ctx for ctx in session_contexts if ctx["id"] in top_contexts]


def build_context_summary(contexts, max_length=500):
    if not contexts:
        return ""
    context_str = "Additional context provided by the user:\n"
    for i, ctx in enumerate(contexts, 1):
        source = (
            ctx.get("filename", "user input") if "filename" in ctx else "user input"
        )
        summary = ctx.get("summary", "No summary available")
        if len(summary) > max_length:
            summary = summary[:max_length] + "..."
        context_str += f"{i}. From {source}: {summary}\n"
    return context_str


def create_new_session():
    session_id = str(uuid.uuid4())
    session = {
        "history": [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant that answers questions about our company. "
                    "Here are key product relationships:\n"
                    "- Product A is used as a component in Product B\n"
                    "- Product C integrates with Product D\n"
                    "- Product B outputs feed Product E"
                ),
            }
        ],
        "contexts": [],
        "created_at": time.time(),
        "last_accessed": time.time(),
        "persistent_memory": {},
        "context_embedding": None,
    }
    save_session(session_id, session)
    return session_id, session


def save_session(session_id, session_data):
    try:
        filepath = os.path.join(SESSION_STORE, f"{session_id}.json")
        with open(filepath, "w") as f:
            for key in ["created_at", "last_accessed"]:
                if key in session_data:
                    session_data[key] = datetime.fromtimestamp(
                        session_data[key]
                    ).isoformat()
            json.dump(session_data, f, indent=2)
    except Exception as e:
        logging.error(f"Error saving session {session_id}: {str(e)}")


def load_session(session_id):
    try:
        filepath = os.path.join(SESSION_STORE, f"{session_id}.json")
        if os.path.exists(filepath):
            with open(filepath, "r") as f:
                session_data = json.load(f)
                for key in ["created_at", "last_accessed"]:
                    if key in session_data:
                        session_data[key] = datetime.fromisoformat(
                            session_data[key]
                        ).timestamp()
                return session_data
    except Exception as e:
        logging.error(f"Error loading session {session_id}: {str(e)}")
    return None


def extract_text_from_file(filepath, filename):
    text = ""
    ext = filename.split(".")[-1].lower()
    try:
        if ext == "txt":
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
        elif ext == "pdf":
            # First try with PyPDF2
            try:
                with open(filepath, "rb") as f:
                    reader = PyPDF2.PdfReader(f)
                    for page in reader.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
                if text.strip():  # If we got text, use it
                    return text.strip()
            except Exception as e:
                logging.error(f"PyPDF2 extraction error: {str(e)}")

            # Fallback to pdfplumber
            try:
                text = ""
                with pdfplumber.open(filepath) as pdf:
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
                return text.strip()
            except Exception as e:
                logging.error(f"PDFplumber extraction error: {str(e)}")
                return f"Error processing PDF: {str(e)}"
        elif ext in ["doc", "docx"]:
            doc = Document(filepath)
            for para in doc.paragraphs:
                text += para.text + "\n"
        elif ext in ["ppt", "pptx"]:
            text = f"PPT file detected: {filename}. Please upload as PDF for full text extraction."
        elif ext in ["jpg", "jpeg", "png"]:
            text = f"Image file detected: {filename}. OCR capability not implemented."
        elif ext == "mp4":
            text = f"Video file detected: {filename}. Video analysis not implemented."
    except Exception as e:
        logging.error(f"Text extraction error: {str(e)}")
        return f"Error processing file: {str(e)}"
    return text.strip()


def chunk_text(text, max_tokens=1000, overlap=100):
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_tokens = 0

    for sentence in sentences:
        sentence_tokens = count_tokens(sentence)

        if current_tokens + sentence_tokens > max_tokens and current_chunk:
            chunks.append(" ".join(current_chunk))
            # Keep overlapping sentences
            overlap_tokens = 0
            while current_chunk and overlap_tokens < overlap:
                removed = current_chunk.pop(0)
                overlap_tokens += count_tokens(removed)
                current_tokens -= count_tokens(removed)

        current_chunk.append(sentence)
        current_tokens += sentence_tokens

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def extract_key_concepts(text, max_concepts=7):
    try:
        stop_words = set(stopwords.words("english"))
        words = word_tokenize(text.lower())
        words = [word for word in words if word.isalnum() and word not in stop_words]
        word_freq = Counter(words)
        most_common = word_freq.most_common(max_concepts * 2)
        concepts = [word for word, freq in most_common if len(word) > 3 and freq > 1]
        return concepts[:max_concepts]
    except:
        words = text.lower().split()
        words = [word for word in words if word.isalnum() and len(word) > 4]
        word_freq = Counter(words)
        return [word for word, freq in word_freq.most_common(max_concepts)]


def generate_memory_key(question, answer):
    content_hash = hashlib.sha256((question + answer).encode()).hexdigest()
    return f"memory_{content_hash[:16]}"


def generate_summary_and_key_concepts(text, api_key, max_length=300):
    client = OpenAI(api_key=api_key)
    if len(text) < 100:
        return text, extract_key_concepts(text)
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a document summarization system. Please provide:\n"
                        "1. A concise summary (2-3 sentences)\n"
                        "2. A comma-separated list of 5-7 key concepts\n"
                        "Format: SUMMARY: [summary text]\nKEY CONCEPTS: [comma separated list]"
                    ),
                },
                {"role": "user", "content": f"Text:\n\n{text[:5000]}"},
            ],
            max_tokens=300,
            temperature=0.2,
        )
        result = response.choices[0].message.content.strip()
        summary = ""
        concepts = []
        summary_match = re.search(
            r"SUMMARY:\s*(.*?)(?=\nKEY CONCEPTS:|$)", result, re.DOTALL | re.IGNORECASE
        )
        if summary_match:
            summary = summary_match.group(1).strip()
        concepts_match = re.search(
            r"KEY CONCEPTS:\s*(.*)", result, re.DOTALL | re.IGNORECASE
        )
        if concepts_match:
            concepts_str = concepts_match.group(1).strip()
            concepts = [c.strip() for c in concepts_str.split(",") if c.strip()]
        if not summary:
            sentences = nltk.sent_tokenize(text)
            summary = " ".join(sentences[:3])[:max_length] + (
                "..." if len(sentences) > 3 else ""
            )
        if not concepts:
            concepts = extract_key_concepts(text)
        return summary, concepts
    except Exception as e:
        logging.error(f"Summary generation failed: {str(e)}")
        sentences = nltk.sent_tokenize(text)
        summary = " ".join(sentences[:3])[:max_length] + (
            "..." if len(sentences) > 3 else ""
        )
        concepts = extract_key_concepts(text)
        return summary, concepts


def generate_vector(text):
    vectorizer = TfidfVectorizer()
    try:
        vector = vectorizer.fit_transform([text]).toarray()[0]
    except:
        vector = np.zeros(100)  # Fallback to zero vector
    return vector


@app.route("/")
def index():
    return render_template("kaggleloc.html")


@app.route("/api/login", methods=["POST"])
def login():
    data = request.get_json()
    user = validate_login(data["account"], data["password"])
    if user:
        return jsonify({"status": "success", "userId": user[0], "account": user[1]})
    return jsonify({"status": "fail"}), 401


@app.route("/api/memories", methods=["GET"])
def get_memories():
    user_id = request.args.get("userId")
    if not user_id:
        return jsonify({"error": "Missing user ID"}), 400
    memories = get_user_memories(user_id)
    return jsonify(
        [
            {
                "id": row[0],
                "question": row[1],
                "answer": row[2],
                "summary": row[3],
                "sources": row[4],
                "created_at": row[5],
            }
            for row in memories
        ]
    )


@app.route("/api/memories/<int:memory_id>", methods=["PUT"])
def update_memory(memory_id):
    data = request.get_json()
    new_answer = data.get("answer", "").strip()
    new_summary = data.get("summary", "").strip()
    new_sources = data.get("sources", "").strip()
    if not new_answer:
        return jsonify({"error": "Answer cannot be empty"}), 400
    try:
        with get_db() as db:
            db.execute(
                """
                UPDATE memories
                SET answer = ?, summary = ?, sources = ?
                WHERE id = ?
            """,
                (new_answer, new_summary, new_sources, memory_id),
            )
            db.commit()
        return jsonify({"status": "success"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/memories/<int:memory_id>", methods=["DELETE"])
def delete_memory(memory_id):
    try:
        with get_db() as db:
            db.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
            db.commit()
        return jsonify({"status": "deleted"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/ask", methods=["POST"])
def handle_question():
    api_key = request.headers.get("Authorization", "").replace("Bearer ", "")
    account_number = request.headers.get("X-Account-Number", "")
    if not api_key:
        return jsonify({"error": "Missing API key"}), 401

    data = request.get_json()
    question = data.get("question", "").strip()
    session_id = data.get("sessionId", "")
    if not question:
        return jsonify({"error": "Empty question"}), 400

    try:
        with sessions_lock:
            if session_id:
                session = sessions.get(session_id)
                if not session:
                    session = load_session(session_id)
                    if session:
                        sessions[session_id] = session
                if not session:
                    session_id, session = create_new_session()
                    sessions[session_id] = session
            else:
                session_id, session = create_new_session()
                sessions[session_id] = session
            session["last_accessed"] = time.time()
            save_session(session_id, session)

        context_docs = rag_processor.retrieve_context(question)
        rag_context = "\n\n".join(context_docs)
        known_response = rag_processor.get_known_relationship(question)
        if known_response:
            return jsonify(
                {
                    "answer": known_response,
                    "insights": "This information comes from our verified product relationship database",
                    "context": [known_response],
                    "sessionId": session_id,
                }
            )

        # Get relevant session contexts
        relevant_contexts = get_relevant_contexts(
            question, session.get("contexts", []), session_id, top_n=3
        )
        context_summary = build_context_summary(relevant_contexts)

        # Build full context
        full_context = []
        if rag_context:
            full_context.append(f"Company Knowledge Base:\n{rag_context}")
        if context_summary:
            full_context.append(context_summary)

        # Prepare messages
        messages = session["history"][:]
        if full_context:
            messages.append({"role": "system", "content": "\n\n".join(full_context)})

        messages.append({"role": "user", "content": question})
        messages = trim_conversation_history(messages)

        client = OpenAI(api_key=api_key)
        chat = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=messages,
            max_tokens=1000,
            temperature=0.3,
        )
        answer = chat.choices[0].message.content.strip()

        # Generate insights
        insight_prompt = (
            "Based on the following question and answer, extract 3 key insights in bullet points:\n\n"
            f"Question: {question}\n\n"
            f"Answer:\n{answer}\n\n"
            "Key Insights:"
        )
        insight_response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "You are an analyst that extracts key insights from Q&A pairs.",
                },
                {"role": "user", "content": insight_prompt},
            ],
            max_tokens=300,
            temperature=0.2,
        )
        insights = insight_response.choices[0].message.content.strip()

        # Generate memory summary
        memory_prompt = (
            "Create a concise summary of the following Q&A pair that captures the key information "
            "for long-term memory storage. Focus on factual information and core concepts:\n\n"
            f"Question: {question}\n\n"
            f"Answer:\n{answer}"
        )
        memory_response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "You are a knowledge distillation system.",
                },
                {"role": "user", "content": memory_prompt},
            ],
            max_tokens=200,
            temperature=0.1,
        )
        memory_summary = memory_response.choices[0].message.content.strip()

        # Generate follow-up questions
        follow_prompt = (
            "Based on the following question and answer, suggest 3 follow-up questions that the user might ask:\n\n"
            f"Question: {question}\n\n"
            f"Answer:\n{answer}\n\n"
            "Follow-up Questions:"
        )
        follow_response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that anticipates user needs.",
                },
                {"role": "user", "content": follow_prompt},
            ],
            max_tokens=200,
            temperature=0.5,
        )
        follow_ups_text = follow_response.choices[0].message.content.strip()
        follow_ups = []
        for line in follow_ups_text.split("\n"):
            if line.strip() and len(follow_ups) < 3:
                clean_line = re.sub(r"^\d+[\.\)]\s*", "", line.strip())
                follow_ups.append(clean_line)

        # Store in database if user is logged in
        if account_number:
            user = get_user_by_account(account_number)
            if user:
                store_memory(
                    user[0], question, answer, memory_summary, "\n".join(context_docs)
                )

        # Update session history
        with sessions_lock:
            session = sessions[session_id]
            session["history"].append({"role": "user", "content": question})
            session["history"].append({"role": "assistant", "content": answer})
            if len(session["history"]) > 21:
                session["history"] = session["history"][:1] + session["history"][-20:]
            session["last_accessed"] = time.time()
            save_session(session_id, session)

        return jsonify(
            {
                "question": question,
                "answer": answer,
                "insights": insights,
                "context": context_docs,
                "sessionId": session_id,
                "follow_ups": follow_ups,
                "memory_summary": memory_summary,
            }
        )

    except AuthenticationError:
        return jsonify({"error": "Invalid API key"}), 401
    except RateLimitError:
        return jsonify({"error": "OpenAI API rate limit exceeded"}), 429
    except BadRequestError as e:
        return jsonify({"error": f"Invalid request: {str(e)}"}), 400
    except OpenAIError as e:
        return jsonify({"error": f"OpenAI error: {str(e)}"}), 500
    except Exception as e:
        logging.exception("Unexpected error")
        return (
            jsonify(
                {
                    "error": str(e),
                    "answer": "I encountered an error processing your question. Please try again.",
                    "insights": "Error: Could not generate insights",
                    "context": [],
                    "sessionId": session_id,
                }
            ),
            500,
        )


@app.route("/api/adjust_answer", methods=["POST"])
def adjust_answer():
    api_key = request.headers.get("Authorization", "").replace("Bearer ", "")
    if not api_key:
        return jsonify({"error": "Missing API key"}), 401

    data = request.get_json()
    original_answer = data.get("answer", "")
    adjustment_type = data.get("type", "shorten")  # 'shorten' or 'elaborate'
    language = data.get("language", "English")

    if not original_answer:
        return jsonify({"error": "No answer provided"}), 400

    try:
        client = OpenAI(api_key=api_key)

        if adjustment_type == "translate":
            prompt = (
                f"Translate the following text to {language}:\n\n" f"{original_answer}"
            )
        elif adjustment_type == "shorten":
            prompt = (
                "Please condense the following text while preserving key information:\n\n"
                f"{original_answer}"
            )
        elif adjustment_type == "elaborate":
            prompt = (
                "Please elaborate on the following text, adding more details and explanations:\n\n"
                f"{original_answer}"
            )
        elif adjustment_type == "reword":
            prompt = (
                "Please rephrase the following text to improve clarity while keeping the meaning unchanged:\n\n"
                f"{original_answer}"
            )
        else:
            return jsonify({"error": "Invalid adjustment type"}), 400

        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful text transformation assistant.",
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=1000,
            temperature=0.5,
        )
        adjusted_answer = response.choices[0].message.content.strip()

        return jsonify(
            {
                "adjusted_answer": adjusted_answer,
                "adjustment_type": adjustment_type,
                "language": language,
            }
        )

    except Exception as e:
        logging.error(f"Error adjusting answer: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/add_context", methods=["POST"])
def add_context():
    api_key = request.headers.get("Authorization", "").replace("Bearer ", "")
    if not api_key:
        return jsonify({"error": "Missing API key"}), 401
    session_id = request.form.get("sessionId", "")
    context_text = request.form.get("context", "").strip()
    if not session_id:
        return jsonify({"error": "Missing session ID"}), 400
    if not context_text:
        return jsonify({"error": "Empty context"}), 400
    try:
        with sessions_lock:
            if session_id in sessions:
                session = sessions[session_id]
            else:
                session = load_session(session_id)
                if not session:
                    session_id, session = create_new_session()
                sessions[session_id] = session

            summary, key_concepts = generate_summary_and_key_concepts(
                context_text, api_key
            )
            context_id = str(uuid.uuid4())

            # Create vector for context
            context_vector = generate_vector(context_text)
            store_context_vector(session_id, context_id, context_vector)

            session["contexts"].append(
                {
                    "id": context_id,
                    "type": "text",
                    "content": context_text,
                    "summary": summary,
                    "key_concepts": key_concepts,
                    "timestamp": time.time(),
                }
            )
            session["history"].append(
                {
                    "role": "system",
                    "content": f"User provided additional context: {summary}",
                }
            )
            session["last_accessed"] = time.time()
            save_session(session_id, session)

        return jsonify(
            {
                "status": "success",
                "sessionId": session_id,
                "summary": summary,
                "key_concepts": key_concepts,
            }
        )
    except Exception as e:
        logging.exception("Error adding context")
        return jsonify({"error": str(e)}), 500


@app.route("/api/upload_context", methods=["POST"])
def upload_context():
    api_key = request.headers.get("Authorization", "").replace("Bearer ", "")
    if not api_key:
        return jsonify({"error": "Missing API key"}), 401
    session_id = request.form.get("sessionId", "")
    if not session_id:
        return jsonify({"error": "Missing session ID"}), 400
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)
        extracted_text = extract_text_from_file(filepath, filename)
        summary = ""
        key_concepts = []
        if extracted_text:
            summary, key_concepts = generate_summary_and_key_concepts(
                extracted_text, api_key
            )
            chunks = chunk_text(extracted_text)

            with sessions_lock:
                if session_id in sessions:
                    session = sessions[session_id]
                else:
                    session = load_session(session_id)
                    if not session:
                        session_id, session = create_new_session()
                    sessions[session_id] = session

                context_id = str(uuid.uuid4())

                # Create vectors for each chunk
                for chunk in chunks:
                    chunk_id = str(uuid.uuid4())
                    chunk_vector = generate_vector(chunk)
                    store_context_vector(session_id, chunk_id, chunk_vector)

                session["contexts"].append(
                    {
                        "id": context_id,
                        "filename": filename,
                        "filepath": filepath,
                        "content": extracted_text,
                        "chunks": chunks,
                        "summary": summary,
                        "key_concepts": key_concepts,
                        "timestamp": time.time(),
                    }
                )
                session["history"].append(
                    {
                        "role": "system",
                        "content": f"User uploaded document: {filename}\nSummary: {summary}",
                    }
                )
                session["last_accessed"] = time.time()
                save_session(session_id, session)

            return jsonify(
                {
                    "status": "success",
                    "filename": filename,
                    "summary": summary,
                    "key_concepts": key_concepts,
                    "sessionId": session_id,
                }
            )
        else:
            return (
                jsonify(
                    {
                        "error": "File processed but no extractable text found",
                        "filename": filename,
                    }
                ),
                400,
            )
    return jsonify({"error": "File type not allowed"}), 400


@app.route("/api/clear_memory", methods=["POST"])
def clear_memory():
    data = request.get_json()
    session_id = data.get("sessionId", "")
    memory_key = data.get("memoryKey", "")
    if not session_id:
        return jsonify({"error": "Missing session ID"}), 400
    with sessions_lock:
        if session_id in sessions:
            session = sessions[session_id]
            if memory_key:
                if memory_key in session.get("persistent_memory", {}):
                    del session["persistent_memory"][memory_key]
                    save_session(session_id, session)
                    return jsonify({"status": "success", "message": "Memory removed"})
                else:
                    return jsonify({"error": "Memory not found"}), 404
            else:
                session["persistent_memory"] = {}
                session["history"] = session["history"][:1]
                session["contexts"] = []
                save_session(session_id, session)
                return jsonify({"status": "success", "message": "All memories cleared"})
        else:
            return jsonify({"error": "Session not found"}), 404


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
