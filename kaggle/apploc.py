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

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# Initialize RAG processor
try:
    rag_processor = RagProcessor()
    logging.info("RAG processor initialized successfully")
except Exception as e:
    logging.error(f"Failed to initialize RAG processor: {str(e)}")
    rag_processor = None


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
    if not question:
        return jsonify({"error": "Empty question"}), 400

    try:
        if not rag_processor:
            raise Exception("RAG processor not initialized")

        # Initialize OpenAI client
        client = OpenAI(api_key=api_key)

        context_docs = rag_processor.retrieve_context(question)
        logging.info(
            f"Retrieved {len(context_docs)} context documents for question: {question}"
        )

        context_str = "\n\n".join(context_docs)
        prompt = (
            "You are a helpful assistant that answers questions about our company. "
            f"Use the following context to answer the question.\n\n"
            f"Context:\n{context_str}\n\n"
            f"Question: {question}\n"
            "Answer:"
        )

        chat = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful company assistant that provides accurate answers based on the given context.",
                },
                {"role": "user", "content": prompt},
            ],
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

        return jsonify(
            {
                "question": question,
                "answer": answer,
                "insights": insights,
                "context": context_docs,
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
                }
            ),
            500,
        )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
