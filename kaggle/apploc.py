# apploc.py
from flask import Flask, render_template, request, jsonify
import subprocess
import json
import uuid
import os

app = Flask(__name__)


@app.route("/", methods=["GET"])
def index():
    session_id = str(uuid.uuid4())
    return render_template("kaggleloc.html", session_id=session_id)


@app.route("/api/ask", methods=["POST"])
def handle_question():
    data = request.json
    question = data.get("question", "")
    if not question.strip():
        return jsonify({"error": "Empty question"}), 400

    print(f"💬 Received question: {question}")

    try:
        # Run rag_pipeline.py with QUESTION environment variable
        result = subprocess.run(
            ["python", "rag_pipeline.py"],
            env={**os.environ, "QUESTION": question},
            capture_output=True,
            text=True,
            timeout=120,
        )
        print("📦 Subprocess output captured.")

        if not os.path.exists("result.json"):
            return jsonify({"error": "No result.json generated"}), 500

        with open("result.json", "r", encoding="utf-8") as f:
            response = json.load(f)

        # Return structured output
        return jsonify(
            {
                "question": question,
                "answer": response.get("answer", ""),
                "insights": response.get("explanation", ""),
                "output": result.stdout,
            }
        )

    except Exception as e:
        print("❌ Error:", e)
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
