import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
from scripts.rag_engine import ask_zooz, load_collection, get_groq_client

load_dotenv()

app = Flask(__name__)

# Warm the expensive local resources during service startup instead of making the
# first real user wait for the multilingual embedding model and ChromaDB to load.
# Creating the Groq client does not send a request or consume tokens.
try:
    load_collection()
    get_groq_client()
    print("ZOOZ RAG warmup complete")
except Exception as exc:
    # Keep the service bootable so Render logs expose the real configuration issue.
    print(f"ZOOZ RAG warmup warning: {exc}")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json() or {}
    question = data.get("question", "").strip()
    if not question:
        return jsonify({"error": "no question"}), 400
    try:
        answer, sources = ask_zooz(question)
        return jsonify({"answer": answer, "sources": sources})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
