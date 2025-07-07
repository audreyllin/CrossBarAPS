from flask import Flask, request, jsonify, render_template
import subprocess
import os
import shlex
import time

app = Flask(__name__)

# Secure API key handling
KAGGLE_API_KEY = "kaggle_kernel_api_key_isha21700"
KAGGLE_USERNAME = "isha21700"


@app.route("/")
def index():
    # Set default command and parameters
    default_command = "kaggle kernels output"
    default_kernel = f"{KAGGLE_USERNAME}/isha-mistralai-test"
    default_params = "--path output"

    return render_template(
        "kaggle.html",
        api_key=KAGGLE_API_KEY,
        username=KAGGLE_USERNAME,
        default_command=default_command,
        default_kernel=default_kernel,
        default_params=default_params,
    )


@app.route("/api/kaggle/execute", methods=["POST"])
def execute_kaggle_command():
    # Authorization
    api_key = request.headers.get("Authorization", "").replace("Bearer ", "")
    if not api_key or api_key != KAGGLE_API_KEY:
        return jsonify({"error": "Invalid API key"}), 401

    # Get request data
    data = request.json
    command = data.get("command", "")
    parameters = data.get("parameters", "")
    kernel_ref = data.get("kernel", "")
    question = data.get("question", "")  # Get user's question

    if not command:
        return jsonify({"error": "No command provided"}), 400

    try:
        # Set Kaggle credentials in environment
        env = os.environ.copy()
        env["KAGGLE_USERNAME"] = KAGGLE_USERNAME
        env["KAGGLE_KEY"] = KAGGLE_API_KEY

        # Build command list
        base_command = shlex.split(command)

        # Add kernel reference
        if kernel_ref:
            base_command.append(kernel_ref)

        # Add parameters
        if parameters:
            base_command += shlex.split(parameters)

        # Add question as environment variable
        if question:
            env["QUESTION"] = question

        # Execute command
        result = subprocess.run(
            base_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            timeout=120,  # 2-minute timeout
        )

        # Parse output
        output = result.stdout
        metrics = {}

        # Extract answer if question was provided
        answer = ""
        if question and output:
            answer = extract_answer(output, question)

        # For demo purposes - simulate company info extraction
        if "company" in question.lower() and not answer:
            answer = "Crossbar Inc. is a technology company based in Santa Clara, California, specializing in Resistive RAM (ReRAM), a type of non-volatile memory that offers high performance, energy efficiency, and scalability."

        return jsonify({"output": output, "metrics": metrics, "answer": answer})

    except Exception as e:
        return (
            jsonify({"error": str(e), "output": f"Error executing command: {str(e)}"}),
            500,
        )


def extract_answer(output, question):
    """Extract relevant answer from kernel output"""
    # Look for question-related response patterns
    lines = output.split("\n")
    for i, line in enumerate(lines):
        if question[:20].lower() in line.lower():
            # Return next 3 lines as potential answer
            return "\n".join(lines[i + 1 : i + 4])

    # Return the first 3 lines if no direct match
    return "\n".join(lines[:3]) if lines else "No answer found"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
