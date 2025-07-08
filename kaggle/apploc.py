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
        rag_path = os.path.join(os.path.dirname(__file__), "rag_pipeline.py")
        result = subprocess.run(
            ["python", rag_path],
            env={**os.environ, "QUESTION": question},
            capture_output=True,
            text=True,
            timeout=120,
        )

        result_path = os.path.join("output", "result.json")
        print(f"📄 Looking for result.json at: {result_path}")
        if not os.path.exists(result_path):
            error_msg = "Missing result.json - Pipeline likely failed"
            print(f"❌ {error_msg}")
            return (
                jsonify(
                    {
                        "error": error_msg,
                        "output": result.stdout,
                        "stderr": result.stderr,
                    }
                ),
                500,
            )

        with open(result_path, "r", encoding="utf-8") as f:
            response = json.load(f)

        return jsonify(
            {
                "question": question,
                "answer": response.get("answer", ""),
                "insights": response.get("explanation", ""),
                "output": result.stdout,
                "stderr": result.stderr,
            }
        )

    except Exception as e:
        print(f"❌ Exception occurred: {e}")
        return (
            jsonify(
                {
                    "error": str(e),
                    "output": result.stdout if "result" in locals() else "",
                    "stderr": result.stderr if "result" in locals() else "",
                }
            ),
            500,
        )


if __name__ == "__main__":
    app.run(debug=True)
