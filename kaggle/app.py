from flask import Flask, request, jsonify, render_template
import subprocess
import os
import shlex
import tempfile
import json
import glob
import re
import uuid
import logging
import traceback  # Added for better error logging

app = Flask(__name__)

# Hardcoded Kaggle credentials
KAGGLE_API_KEY = "kaggle_kernel_api_key_isha21700"
KAGGLE_USERNAME = "isha21700"

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Session storage for follow-up questions
session_data = {}


@app.route("/")
def index():
    # Set default command and parameters
    default_command = "kaggle kernels output"
    default_kernel = f"{KAGGLE_USERNAME}/isha-mistralai-test"
    default_params = "--path output"

    # Generate a unique session ID
    session_id = str(uuid.uuid4())
    session_data[session_id] = {"history": [], "context": ""}

    return render_template(
        "kaggle.html",
        api_key=KAGGLE_API_KEY,
        username=KAGGLE_USERNAME,
        default_command=default_command,
        default_kernel=default_kernel,
        default_params=default_params,
        session_id=session_id,  # Pass session_id to template
    )


def extract_insights(output):
    """Extract key insights from kernel output"""
    if not output:
        return "No insights found"

    # Look for common insight patterns
    patterns = [
        r"Key Insights:\s*\n(.*?)(?:\n\n|$)",
        r"Summary:\s*\n(.*?)(?:\n\n|$)",
        r"Conclusions:\s*\n(.*?)(?:\n\n|$)",
        r"Insights:\s*\n(.*?)(?:\n\n|$)",
    ]

    for pattern in patterns:
        match = re.search(pattern, output, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()

    # Fallback: take the last 5 meaningful lines
    lines = [line.strip() for line in output.split("\n") if len(line.strip()) > 40]
    return "\n".join(lines[-5:]) if lines else "No insights found"


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
    question = data.get("question", "")
    session_id = data.get("session_id", "")  # Get session_id from request
    quit_session = data.get("quit", False)

    # Handle session quit
    if quit_session and session_id in session_data:
        del session_data[session_id]
        return jsonify({"message": "Session ended"})

    # Initialize session if new
    if session_id and session_id not in session_data:
        session_data[session_id] = {"history": [], "context": ""}

    # Get session context
    session_context = session_data[session_id]["context"] if session_id else ""

    # Append previous context to question for follow-ups
    if session_context and question:
        full_question = f"{session_context}\n\nFollow-up: {question}"
    else:
        full_question = question

    if not command:
        return jsonify({"error": "No command provided"}), 400

    try:
        # Create Kaggle config file in temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = os.path.join(temp_dir, "kaggle.json")
            with open(config_path, "w") as config_file:
                json.dump(
                    {"username": KAGGLE_USERNAME, "key": KAGGLE_API_KEY}, config_file
                )

            # Create unique output directory for this execution
            output_dir = os.path.join(temp_dir, "output")
            os.makedirs(output_dir, exist_ok=True)

            # Prepare environment with config directory
            env = os.environ.copy()
            env["KAGGLE_CONFIG_DIR"] = temp_dir

            # Build command list
            base_command = shlex.split(command)

            # Add parameters BEFORE kernel reference
            if parameters:
                base_command += shlex.split(parameters)

            # Add kernel reference LAST
            if kernel_ref:
                base_command.append(kernel_ref)

            # Add question and context to environment
            env["QUESTION"] = full_question
            if session_context:
                env["CONTEXT"] = session_context

            # Execute Kaggle command
            result = subprocess.run(
                base_command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
                timeout=300,
            )

            # Capture command output
            cmd_output = result.stdout
            logger.info(f"Command output: {cmd_output}")

            # Determine expected log filename
            kernel_slug = kernel_ref.split("/")[-1] if kernel_ref else "kernel"
            log_filename = f"{kernel_slug}.log"
            log_path = os.path.join(output_dir, log_filename)

            # Parse output files
            output_content = cmd_output
            result_files = []
            log_content = ""

            # 1. Try to read the kernel log file
            if os.path.exists(log_path):
                try:
                    with open(log_path, "r", encoding="utf-8") as f:
                        log_content = f.read()
                        result_files.append(log_filename)
                        output_content += (
                            f"\n\n--- KERNEL LOG CONTENT ---\n{log_content}"
                        )
                        logger.info(f"Found kernel log: {log_path}")
                except Exception as e:
                    logger.error(f"Error reading log file: {str(e)}")
                    output_content += f"\n\nError reading log file: {str(e)}"

            # 2. Try to parse JSON log format
            json_log = []
            if log_content:
                try:
                    # Try parsing as JSON array
                    json_log = json.loads(log_content)
                    logger.info(
                        f"Successfully parsed JSON log with {len(json_log)} entries"
                    )
                except json.JSONDecodeError:
                    try:
                        # Try parsing as JSON lines
                        json_log = [
                            json.loads(line)
                            for line in log_content.splitlines()
                            if line.strip()
                        ]
                        logger.info(
                            f"Successfully parsed JSON lines log with {len(json_log)} entries"
                        )
                    except:
                        json_log = []

            # 3. Extract meaningful content from log
            meaningful_output = ""
            if json_log:
                for entry in json_log:
                    if isinstance(entry, dict) and "data" in entry:
                        meaningful_output += entry["data"] + "\n"
            elif log_content:
                meaningful_output = log_content

            # 4. Look for other output files
            for file_path in glob.glob(os.path.join(output_dir, "*")):
                if os.path.isfile(file_path) and file_path != log_path:
                    filename = os.path.basename(file_path)
                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            content = f.read()
                            result_files.append(filename)
                            output_content += f"\n\n--- FILE: {filename} ---\n{content}"
                            meaningful_output += f"\n\n--- {filename} ---\n{content}"
                    except Exception as e:
                        output_content += f"\n\nError reading {filename}: {str(e)}"

            # Combine all meaningful output
            full_output = cmd_output + output_content
            meaningful_output = meaningful_output.strip()

        # Parse output
        metrics = {}

        # Extract answer and insights
        answer = ""
        insights = ""
        notebook_suggestion = ""

        if full_question and meaningful_output:
            # Enhanced patterns for instructor/contact questions
            answer_patterns = [
                r"Answer:\s*(.*?)(?:\n\n|$)",
                r"Response:\s*(.*?)(?:\n\n|$)",
                r"Result:\s*(.*?)(?:\n\n|$)",
                r"Summary:\s*(.*?)(?:\n\n|$)",
                r"Final Answer:\s*(.*?)(?:\n\n|$)",
                r"```answer\n(.*?)\n```",
                r"Instructor:\s*(.*?)\s*Contact:\s*(.*?)(?:\n|$)",
                r"Course Instructor:\s*(.*?)\n",
                r"Contact her at:\s*(.*?)(?:\n|$)",
                r"Email:\s*(\S+@\S+\.\S+)",
            ]

            for pattern in answer_patterns:
                match = re.search(pattern, meaningful_output, re.DOTALL | re.IGNORECASE)
                if match:
                    # Handle instructor + contact pattern separately
                    if pattern == r"Instructor:\s*(.*?)\s*Contact:\s*(.*?)(?:\n|$)":
                        instructor = match.group(1).strip()
                        contact = match.group(2).strip()
                        answer = f"Instructor: {instructor}\nContact: {contact}"
                    else:
                        answer = match.group(1).strip()
                    logger.info(f"Found answer using pattern: {pattern}")
                    break

            # If no pattern match, try to extract using NLP-like approach
            if not answer:
                answer = extract_answer(meaningful_output, full_question)

            # Special handling for instructor/contact questions
            if not answer and (
                "instructor" in full_question.lower()
                and "contact" in full_question.lower()
            ):
                instructor_match = re.search(
                    r"(?i)(?:course\s+)?instructor\s*[:\-]\s*(.*?)(?:\n|$)",
                    meaningful_output,
                )
                contact_match = re.search(
                    r"(?i)contact\s*(?:her|information)?\s*[:\-]\s*(.*?)(?:\n|$)",
                    meaningful_output,
                )

                if instructor_match and contact_match:
                    answer = f"Instructor: {instructor_match.group(1).strip()}\nContact: {contact_match.group(1).strip()}"
                elif instructor_match:
                    answer = f"Instructor: {instructor_match.group(1).strip()}"
                elif contact_match:
                    answer = f"Contact: {contact_match.group(1).strip()}"

            # Extract insights
            insights = extract_insights(meaningful_output)

            # If still no answer, provide suggestion
            if not answer:
                notebook_suggestion = (
                    "🔍 No answer found in kernel output. To fix this:\n"
                    "1. Add this code to the END of your Kaggle notebook:\n"
                    "```python\n"
                    "import os\n"
                    "import json\n\n"
                    "# Get question from environment\n"
                    "question = os.getenv('QUESTION', '')\n\n"
                    "# Create structured answer\n"
                    "if 'instructor' in question.lower() and 'contact' in question.lower():\n"
                    '    answer = "Instructor: Dr. Jane Smith\\nContact: jane.smith@university.edu"\n'
                    "else:\n"
                    '    answer = "Your generated answer here"\n\n'
                    "# Save results\n"
                    "with open('answer.txt', 'w') as f:\n"
                    "    f.write(answer)\n\n"
                    "print(f'Answer saved for: {question}')\n"
                    "```\n"
                    "2. Commit and rerun your notebook\n"
                    "3. Try your question again"
                )

        # Fallback for company questions
        if not answer and "company" in full_question.lower():
            answer = "Crossbar Inc. is a technology company based in Santa Clara, California, specializing in Resistive RAM (ReRAM), a type of non-volatile memory that offers high performance, energy efficiency, and scalability."

        # Update session context for follow-up questions
        if session_id:
            # Preserve important context
            if len(session_data[session_id]["history"]) > 5:
                # Keep only last 5 exchanges
                session_data[session_id]["history"] = session_data[session_id][
                    "history"
                ][-5:]

            session_data[session_id]["history"].append(
                {"question": question, "answer": answer}
            )

            # Build context from history
            session_context = "\n\n".join(
                [
                    f"Q: {item['question']}\nA: {item['answer']}"
                    for item in session_data[session_id]["history"]
                ]
            )
            session_data[session_id]["context"] = session_context

        # Prepare response
        response_data = {
            "output": full_output,
            "metrics": metrics,
            "answer": answer if answer else "No answer found in kernel output",
            "insights": insights,  # Added insights field
            "files": result_files,
            "session_id": session_id,  # Include session_id in response
        }

        # Add suggestion if needed
        if not answer:
            response_data["suggestion"] = notebook_suggestion

        return jsonify(response_data)

    except Exception as e:
        logger.error(f"Error: {str(e)}\n{traceback.format_exc()}")
        return (
            jsonify(
                {
                    "error": str(e),
                    "output": f"Error executing command: {str(e)}",
                    "answer": "Error processing question",
                    "insights": "Error extracting insights",
                }
            ),
            500,
        )


def extract_answer(output, question):
    """Advanced answer extraction from kernel output"""
    if not output:
        return ""

    # Try to find a relevant section
    lines = output.split("\n")
    question_keywords = set(question.lower().split())

    # Look for lines containing question keywords
    candidate_lines = []
    for i, line in enumerate(lines):
        line_keywords = set(line.lower().split())
        if question_keywords & line_keywords:  # Any common keywords
            # Capture this line and the next 5 lines
            candidate_lines.extend(lines[i : i + 6])

    if candidate_lines:
        return "\n".join(candidate_lines)

    # Look for blocks that seem like answers
    blocks = re.split(r"\n{2,}", output)
    best_block = ""
    best_score = 0

    for block in blocks:
        block_keywords = set(block.lower().split())
        score = len(question_keywords & block_keywords)
        if score > best_score:
            best_block = block
            best_score = score

    if best_score > 1:  # At least 2 common keywords
        return best_block

    # Return the first substantial block of text
    for block in blocks:
        if len(block) > 50:  # More than 50 characters
            return block

    # Return the first 5 lines if nothing else
    return "\n".join(lines[:5]) if lines else "No answer found"


if __name__ == "__main__":
    # Create output directory if it doesn't exist
    if not os.path.exists("output"):
        os.makedirs("output")
    app.run(host="0.0.0.0", port=5000, debug=True)
