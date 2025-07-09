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

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# Initialize RAG processor
try:
    rag_processor = RagProcessor()
    logging.info("RAG processor initialized successfully")
except Exception as e:
    logging.error(f"Failed to initialize RAG processor: {str(e)}")
    rag_processor = None

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


def create_new_session():
    """Create a new session with initial system prompt"""
    return {
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
    }


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
            if session_id not in sessions:
                session_id = str(uuid.uuid4())
                sessions[session_id] = create_new_session()
            session = sessions[session_id]
            session["last_accessed"] = time.time()

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

        # Update session history
        with sessions_lock:
            session["history"].append({"role": "user", "content": question})
            session["history"].append({"role": "assistant", "content": answer})
            # Keep only the last 10 exchanges
            if len(session["history"]) > 21:  # 1 system + 10 Q/A pairs
                session["history"] = session["history"][:1] + session["history"][-20:]

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
            if session_id not in sessions:
                sessions[session_id] = create_new_session()
            session = sessions[session_id]

            # Add context to session
            session["contexts"].append(context_text)

            # Also add to history as a system message
            session["history"].append(
                {
                    "role": "system",
                    "content": f"Additional context provided by user: {context_text}",
                }
            )

            session["last_accessed"] = time.time()

        return jsonify({"status": "success", "sessionId": session_id})

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

        # Add to session context
        with sessions_lock:
            if session_id not in sessions:
                sessions[session_id] = create_new_session()
            session = sessions[session_id]
            session["contexts"].append(f"Uploaded file: {filename}")

            # Add to history as a system message
            session["history"].append(
                {"role": "system", "content": f"User uploaded file: {filename}"}
            )

            session["last_accessed"] = time.time()

        return jsonify(
            {"status": "success", "filename": filename, "sessionId": session_id}
        )

    return jsonify({"error": "File type not allowed"}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
