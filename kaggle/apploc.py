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
from docx import Document
import hashlib
from datetime import datetime

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

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
    """Trim conversation history to stay within token limits"""
    total_tokens = 0
    trimmed_history = []

    # Keep system message always
    if history and history[0]["role"] == "system":
        total_tokens += count_tokens(history[0]["content"])
        trimmed_history.append(history[0])

    # Add messages from newest to oldest until token limit reached
    for msg in reversed(history[1:]):
        msg_tokens = count_tokens(msg["content"])
        if total_tokens + msg_tokens > max_tokens:
            break
        total_tokens += msg_tokens
        trimmed_history.insert(1, msg)  # Insert after system message

    return trimmed_history


def get_relevant_contexts(question, session_contexts):
    """Find contexts most relevant to the current question"""
    relevant = []
    if not question or not session_contexts:
        return relevant

    # Preprocess question
    q_lower = question.lower()
    q_keywords = set(q_lower.split())

    for ctx in session_contexts:
        # Skip contexts without summaries
        if "summary" not in ctx:
            continue

        # Create combined content string for matching
        content = (ctx.get("summary", "") + " " + ctx.get("content", "")).lower()

        # Simple keyword matching
        if any(keyword in content for keyword in q_keywords):
            relevant.append(ctx)

    return relevant


def build_context_summary(contexts):
    """Create a concise summary of all relevant contexts"""
    if not contexts:
        return ""

    context_str = "Additional context provided by the user:\n"
    for i, ctx in enumerate(contexts, 1):
        source = (
            ctx.get("filename", "user input") if "filename" in ctx else "user input"
        )
        summary = ctx.get("summary", "No summary available")
        context_str += f"{i}. From {source}: {summary}\n"
    return context_str


def create_new_session():
    """Create a new session with initial system prompt"""
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
        "context_embedding": None,  # Placeholder for future embedding-based matching
    }
    save_session(session_id, session)
    return session_id, session


def save_session(session_id, session_data):
    """Save session to persistent storage"""
    try:
        filepath = os.path.join(SESSION_STORE, f"{session_id}.json")
        with open(filepath, "w") as f:
            # Convert datetime objects to strings
            for key in ["created_at", "last_accessed"]:
                if key in session_data:
                    session_data[key] = datetime.fromtimestamp(
                        session_data[key]
                    ).isoformat()

            json.dump(session_data, f, indent=2)
    except Exception as e:
        logging.error(f"Error saving session {session_id}: {str(e)}")


def load_session(session_id):
    """Load session from persistent storage"""
    try:
        filepath = os.path.join(SESSION_STORE, f"{session_id}.json")
        if os.path.exists(filepath):
            with open(filepath, "r") as f:
                session_data = json.load(f)
                # Convert string timestamps back to floats
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
    """Extract text from supported file types"""
    text = ""
    ext = filename.split(".")[-1].lower()

    try:
        if ext == "txt":
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()

        elif ext == "pdf":
            with open(filepath, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"

        elif ext in ["doc", "docx"]:
            doc = Document(filepath)
            for para in doc.paragraphs:
                text += para.text + "\n"

        elif ext in ["ppt", "pptx"]:
            # PPT conversion requires additional libraries
            text = f"PPT file detected: {filename}. Please upload as PDF for full text extraction."

        elif ext in ["jpg", "jpeg", "png"]:
            text = f"Image file detected: {filename}. OCR capability not implemented."

        elif ext == "mp4":
            text = f"Video file detected: {filename}. Video analysis not implemented."

    except Exception as e:
        logging.error(f"Text extraction error: {str(e)}")
        return f"Error processing file: {str(e)}"

    return text.strip()


def chunk_text(text, max_tokens=1000):
    """Split text into manageable chunks"""
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0

    for word in words:
        word_tokens = count_tokens(word)
        if current_length + word_tokens > max_tokens and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_length = 0
        current_chunk.append(word)
        current_length += word_tokens

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def generate_memory_key(question, answer):
    """Generate a unique key for storing memories"""
    content_hash = hashlib.sha256((question + answer).encode()).hexdigest()
    return f"memory_{content_hash[:16]}"


@app.route("/")
def index():
    return render_template("kaggleloc.html")


@app.route("/api/ask", methods=["POST"])
def handle_question():
    api_key = request.headers.get("Authorization", "").replace("Bearer ", "")
    if not api_key:
        return jsonify({"error": "Missing API key"}), 401

    data = request.get_json()
    question = data.get("question", "").strip()
    session_id = data.get("sessionId", "")
    if not question:
        return jsonify({"error": "Empty question"}), 400

    try:
        # Get or create session
        with sessions_lock:
            if session_id:
                # Try to load from memory first
                session = sessions.get(session_id)
                if not session:
                    # Try to load from persistent storage
                    session = load_session(session_id)
                    if session:
                        sessions[session_id] = session

                if not session:
                    # Create new session
                    session_id, session = create_new_session()
                    sessions[session_id] = session
            else:
                # Create new session
                session_id, session = create_new_session()
                sessions[session_id] = session

            session["last_accessed"] = time.time()
            save_session(session_id, session)

        # Retrieve RAG context
        context_docs = rag_processor.retrieve_context(question)
        rag_context = "\n\n".join(context_docs)

        # Check for known relationships first
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

        # Prepare messages for OpenAI
        messages = session["history"][:]  # Copy session history

        # Add persistent memories if relevant
        memory_context = ""
        for memory_key, memory_content in session.get("persistent_memory", {}).items():
            if question.lower() in memory_content["question"].lower():
                memory_context += f"\nPrevious discussion about this topic: {memory_content['summary']}"

        if memory_context:
            messages.append({"role": "system", "content": memory_context})

        # Get relevant session contexts
        relevant_contexts = get_relevant_contexts(question, session.get("contexts", []))
        context_summary = build_context_summary(relevant_contexts)

        # Add context summary as a single system message
        if context_summary:
            messages.append({"role": "system", "content": context_summary})

        # Add RAG context as a system message
        if rag_context:
            messages.append(
                {
                    "role": "system",
                    "content": f"Document context for this question:\n{rag_context}",
                }
            )

        # Add user's question
        messages.append({"role": "user", "content": question})

        # Log messages for debugging
        logging.info(f"Messages being sent to OpenAI: {json.dumps(messages, indent=2)}")
        logging.info(
            f"Total tokens: {sum(count_tokens(m['content']) for m in messages)}"
        )

        # Trim history to avoid exceeding token limits
        messages = trim_conversation_history(messages)

        # Initialize OpenAI client
        client = OpenAI(api_key=api_key)

        # Get answer
        chat = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=500,
            temperature=0.3,
        )

        answer = chat.choices[0].message.content.strip()
        logging.info(f"Received answer: {answer}")

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

        # Store in persistent memory
        memory_key = generate_memory_key(question, answer)
        with sessions_lock:
            session = sessions[session_id]
            session["persistent_memory"][memory_key] = {
                "question": question,
                "answer": answer,
                "summary": memory_summary,
                "timestamp": time.time(),
            }
            session["last_accessed"] = time.time()
            save_session(session_id, session)

        # Update session history
        with sessions_lock:
            session = sessions[session_id]
            session["history"].append({"role": "user", "content": question})
            session["history"].append({"role": "assistant", "content": answer})
            # Keep only the last 10 exchanges
            if len(session["history"]) > 21:  # 1 system + 10 Q/A pairs
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
        # Get or create session
        with sessions_lock:
            if session_id in sessions:
                session = sessions[session_id]
            else:
                session = load_session(session_id)
                if not session:
                    session_id, session = create_new_session()
                sessions[session_id] = session

            # Generate more focused summary for context
            client = OpenAI(api_key=api_key)
            summary_prompt = (
                "Create a concise summary (1-2 sentences) of the following text focusing on key facts "
                "that would be relevant for answering future questions:\n\n"
                f"{context_text}"
            )

            summary_response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a knowledge distillation system. Extract only the most important information.",
                    },
                    {"role": "user", "content": summary_prompt},
                ],
                max_tokens=100,  # More concise summary
                temperature=0.1,
            )
            summary = summary_response.choices[0].message.content.strip()
            logging.info(f"Generated context summary: {summary}")

            # Add context to session
            context_id = str(uuid.uuid4())
            session["contexts"].append(
                {
                    "id": context_id,
                    "type": "text",
                    "content": context_text,
                    "summary": summary,
                    "timestamp": time.time(),
                }
            )

            # Also add to history as a system message
            session["history"].append(
                {
                    "role": "system",
                    "content": f"User provided additional context: {summary}",
                }
            )

            session["last_accessed"] = time.time()
            save_session(session_id, session)

        return jsonify(
            {"status": "success", "sessionId": session_id, "summary": summary}
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

    # Check if the post request has the file part
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        # Extract and process file content
        extracted_text = extract_text_from_file(filepath, filename)
        summary = ""

        if extracted_text:
            # Generate more focused summary for the document
            try:
                client = OpenAI(api_key=api_key)
                summary_prompt = (
                    "Create a concise summary (2-3 sentences) of the following document "
                    "focusing on key information that would be relevant for answering questions:\n\n"
                    f"{extracted_text[:5000]}"  # Use less text for summary
                )

                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a document summarization system. Extract only the most important information.",
                        },
                        {"role": "user", "content": summary_prompt},
                    ],
                    max_tokens=150,  # More concise summary
                    temperature=0.2,
                )
                summary = response.choices[0].message.content.strip()
                logging.info(f"Generated document summary: {summary}")
            except Exception as e:
                logging.error(f"Summary generation failed: {str(e)}")
                summary = "Document processed but summary failed"

            # Chunk text for potential future use
            chunks = chunk_text(extracted_text)

            with sessions_lock:
                # Get or create session
                if session_id in sessions:
                    session = sessions[session_id]
                else:
                    session = load_session(session_id)
                    if not session:
                        session_id, session = create_new_session()
                    sessions[session_id] = session

                # Store document information
                context_id = str(uuid.uuid4())
                session["contexts"].append(
                    {
                        "id": context_id,
                        "filename": filename,
                        "filepath": filepath,
                        "content": extracted_text,
                        "chunks": chunks,
                        "summary": summary,
                        "timestamp": time.time(),
                    }
                )

                # Add to history
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
                # Remove specific memory
                if memory_key in session.get("persistent_memory", {}):
                    del session["persistent_memory"][memory_key]
                    save_session(session_id, session)
                    return jsonify({"status": "success", "message": "Memory removed"})
                else:
                    return jsonify({"error": "Memory not found"}), 404
            else:
                # Clear all non-core memories
                session["persistent_memory"] = {}
                # Keep only the initial system message
                session["history"] = session["history"][:1]
                session["contexts"] = []
                save_session(session_id, session)
                return jsonify({"status": "success", "message": "All memories cleared"})
        else:
            return jsonify({"error": "Session not found"}), 404


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
