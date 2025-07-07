from flask import Flask, request, jsonify, render_template
import subprocess
import os

app = Flask(__name__)

# Secure API key handling
KAGGLE_API_KEY = (
    "kaggle_kernel_api_key_isha21700"  # Still hardcoded but not exposed in validation
)
KAGGLE_USERNAME = "isha21700"


@app.route("/")
def index():
    return render_template(
        "index.html", api_key=KAGGLE_API_KEY, username=KAGGLE_USERNAME
    )


@app.route("/api/kaggle/execute", methods=["POST"])
def execute_kaggle_command():
    # Get API key from authorization header
    api_key = request.headers.get("Authorization", "").replace("Bearer ", "")

    # Validate API key against our secure constant
    if not api_key or api_key != KAGGLE_API_KEY:
        return jsonify({"error": "Invalid API key"}), 401

    # Get request data
    data = request.json
    command = data.get("command", "")
    parameters = data.get("parameters", "")
    kernel_ref = data.get("kernel", "")

    if not command:
        return jsonify({"error": "No command provided"}), 400

    try:
        # Set Kaggle credentials
        os.environ["KAGGLE_USERNAME"] = KAGGLE_USERNAME
        os.environ["KAGGLE_KEY"] = KAGGLE_API_KEY  # Use our constant, not user input

        # Execute Kaggle command
        process = subprocess.Popen(
            command.split() + (parameters.split() if parameters else []),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        output = []
        while True:
            line = process.stdout.readline()
            if not line and process.poll() is not None:
                break
            if line:
                output.append(line.strip())

        # Parse metrics if available
        metrics = {}
        for line in output:
            if "Accuracy:" in line:
                metrics["accuracy"] = float(line.split(":")[1].strip().replace("%", ""))

        return jsonify({"output": "\n".join(output), "metrics": metrics})

    except Exception as e:
        return (
            jsonify({"error": str(e), "output": f"Error executing command: {str(e)}"}),
            500,
        )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
