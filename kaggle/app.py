from flask import Flask, request, jsonify, render_template
import subprocess
import os

app = Flask(__name__)

# Secure API key handling - consider using environment variables for production
KAGGLE_API_KEY = "kaggle_kernel_api_key_isha21700"
KAGGLE_USERNAME = "isha21700"


@app.route("/")
def index():
    return render_template(
        "kaggle.html", api_key=KAGGLE_API_KEY, username=KAGGLE_USERNAME
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
        # Create a copy of environment variables
        env = os.environ.copy()
        env["KAGGLE_USERNAME"] = KAGGLE_USERNAME
        env["KAGGLE_KEY"] = KAGGLE_API_KEY

        # Build command list
        cmd_list = command.split()
        if parameters:
            cmd_list += parameters.split()

        # Execute Kaggle command
        process = subprocess.Popen(
            cmd_list,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
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
    app.run(host="0.0.0.0", port=5000, debug=True)
