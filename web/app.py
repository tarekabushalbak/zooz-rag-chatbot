import sys
import os
from collections import OrderedDict
from uuid import uuid4

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
from scripts.rag_engine import ask_zooz

load_dotenv()

app = Flask(__name__)

# Lightweight in-memory conversation memory.
# It is intentionally short and per browser session so follow-up questions can
# understand context without changing the RAG knowledge base or storing secrets.
CONVERSATION_COOKIE = "zooz_conversation"
MAX_CONVERSATIONS = 200
MAX_TURNS = 2
MAX_ANSWER_CONTEXT_CHARS = 700
_conversations = OrderedDict()


def _valid_conversation_id(value):
    if not value or len(value) != 32:
        return False
    return all(ch in "0123456789abcdef" for ch in value.lower())


def _conversation_id():
    value = request.cookies.get(CONVERSATION_COOKIE, "")
    if _valid_conversation_id(value):
        return value
    return uuid4().hex


def _get_history(conversation_id):
    history = _conversations.get(conversation_id, [])
    if conversation_id in _conversations:
        _conversations.move_to_end(conversation_id)
    return history


def _remember_turn(conversation_id, question, answer):
    history = list(_conversations.get(conversation_id, []))
    history.append({
        "question": question.strip(),
        "answer": (answer or "").strip()[:MAX_ANSWER_CONTEXT_CHARS],
    })
    _conversations[conversation_id] = history[-MAX_TURNS:]
    _conversations.move_to_end(conversation_id)

    while len(_conversations) > MAX_CONVERSATIONS:
        _conversations.popitem(last=False)


def _looks_like_follow_up(question):
    q = " ".join((question or "").strip().lower().split())
    if not q:
        return False

    follow_up_prefixes = (
        "ומה לגבי",
        "ומה עם",
        "ומה בנושא",
        "ולגבי",
        "ומה",
        "ואיך",
        "ולמה",
        "ואילו",
        "אז מה",
        "אז איך",
        "ומה מהם",
    )
    if q.startswith(follow_up_prefixes):
        return True

    # Short questions containing references such as "זה", "אותו" or "הראשון"
    # are usually dependent on the previous turn.
    if len(q) <= 55:
        reference_terms = (
            "זה",
            "זאת",
            "אותו",
            "אותה",
            "אותם",
            "אותן",
            "הראשון",
            "השני",
            "השלישי",
            "כזה",
            "כאלה",
            "שם",
        )
        words = set(q.replace("?", "").replace(",", "").split())
        if any(term in words for term in reference_terms):
            return True

    return False


def _contextualize_question(question, history):
    """Add only a small amount of recent conversation context for follow-ups.

    The existing RAG engine still retrieves evidence from ZOOZ sources and keeps
    its hallucination safeguards. Independent questions are sent unchanged.
    """
    if not history or not _looks_like_follow_up(question):
        return question

    previous = history[-1]
    previous_question = previous.get("question", "").strip()
    previous_answer = previous.get("answer", "").strip()

    return (
        f"שאלת המשך נוכחית: {question}\n"
        f"הקשר מהתור הקודם באותה שיחה:\n"
        f"שאלה קודמת: {previous_question}\n"
        f"תשובה קודמת: {previous_answer}\n"
        "ענה על שאלת ההמשך בלבד, והסתמך על מקורות ZOOZ שהמערכת מאחזרת."
    )


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

    conversation_id = _conversation_id()
    history = _get_history(conversation_id)
    rag_question = _contextualize_question(question, history)

    try:
        answer, sources = ask_zooz(rag_question)

        # Do not make a temporary technical failure part of the conversation memory.
        if not (answer or "").startswith("אירעה שגיאה טכנית"):
            _remember_turn(conversation_id, question, answer)

        response = jsonify({"answer": answer, "sources": sources})
        response.set_cookie(
            CONVERSATION_COOKIE,
            conversation_id,
            max_age=8 * 60 * 60,
            httponly=True,
            samesite="Lax",
            secure=request.is_secure,
        )
        return response

    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)
