import sys
import os
import re
from collections import OrderedDict
from uuid import uuid4

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
from scripts.rag_engine import ask_zooz

load_dotenv()

app = Flask(__name__)

# Lightweight in-memory conversation memory.
# The memory is deliberately conservative: only clear follow-up questions are
# contextualized. Independent questions are always sent to the RAG unchanged.
CONVERSATION_COOKIE = "zooz_conversation"
MAX_CONVERSATIONS = 200
MAX_TURNS = 1
MAX_ANSWER_CONTEXT_CHARS = 350
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
    """Return True only for questions that clearly depend on prior context.

    We intentionally avoid broad prefixes such as "ומה", "ואיך" or "למה"
    because they can also start completely new questions.
    """
    q = " ".join((question or "").strip().lower().split())
    if not q:
        return False

    explicit_prefixes = (
        "ומה לגבי",
        "ומה עם",
        "ומה בנושא",
        "ולגבי",
        "ומה מהם",
        "ומה מהן",
        "איזה מהם",
        "איזו מהן",
        "מי מהם",
        "מי מהן",
        "מה לגבי זה",
        "מה לגבי זאת",
        "מה עם זה",
        "מה עם זאת",
        "תסביר את זה",
        "תסביר את זאת",
        "תסביר על זה",
        "תסביר על זאת",
        "אפשר להרחיב על זה",
        "אפשר להרחיב על זאת",
        "אפשר לפרט על זה",
        "אפשר לפרט על זאת",
        "מה האפשרות הראשונה",
        "מה האפשרות השנייה",
        "ומה האפשרות הראשונה",
        "ומה האפשרות השנייה",
        "ומה עוד לגבי",
        "ומה עוד על",
    )

    if q.startswith(explicit_prefixes):
        return True

    # Very short elliptical follow-ups that are meaningless without prior context.
    exact_follow_ups = {
        "למה?",
        "למה",
        "איך?",
        "איך",
        "ומה עוד?",
        "ומה עוד",
        "תפרט",
        "תפרט יותר",
        "אפשר להרחיב?",
        "אפשר להרחיב",
    }

    if q in exact_follow_ups:
        return True

    return False


def _contextualize_question(question, history):
    """Attach minimal prior context only when the question is a clear follow-up."""
    if not history or not _looks_like_follow_up(question):
        return question

    previous = history[-1]
    previous_question = previous.get("question", "").strip()
    previous_answer = previous.get("answer", "").strip()

    return (
        f"שאלת המשך נוכחית: {question}\n"
        f"הקשר קצר מהתור הקודם באותה שיחה:\n"
        f"שאלה קודמת: {previous_question}\n"
        f"תשובה קודמת: {previous_answer}\n"
        "השתמש בהקשר רק כדי להבין למה המשתמש מתייחס. "
        "את התשובה עצמה יש לבסס רק על מקורות ZOOZ שהמערכת מאחזרת."
    )


def _clean_answer_formatting(answer):
    """Normalize lightweight Markdown/HTML artifacts before sending to the UI.

    The model occasionally emits Markdown tables, duplicate bullets, literal
    <br> tags, or unmatched single asterisks. The web client intentionally
    supports only simple bold text and line breaks, so normalize those artifacts
    without changing the answer facts.
    """
    text = str(answer or "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)

    cleaned_lines = []
    for raw_line in text.split("\n"):
        line = raw_line.strip()

        if not line:
            if cleaned_lines and cleaned_lines[-1] != "":
                cleaned_lines.append("")
            continue

        # Remove Markdown table separator rows such as |---|---|.
        table_core = line.strip("|").strip()
        if "|" in line and table_core and re.fullmatch(r"[:\-\s|]+", table_core):
            continue

        # Convert Markdown table rows into readable text/bullets.
        if "|" in line:
            cells = [cell.strip() for cell in line.strip("|").split("|") if cell.strip()]
            if len(cells) >= 2:
                # Replace generic Markdown table headers with a natural section title.
                if any(label in cells[0] for label in ("תחום", "נושא", "קטגוריה")):
                    cleaned_lines.append("**השירותים המתאימים:**")
                else:
                    # Remove only real bullet markers. Keep Markdown **bold** intact.
                    normalized_cells = [
                        re.sub(r"^(?:[•\-]\s*)+", "", cell).strip()
                        for cell in cells
                    ]
                    cleaned_lines.append("• " + " — ".join(normalized_cells))
                continue

        # Remove a leftover table pipe at the beginning/end of a wrapped line.
        line = line.strip("|").strip()

        # Collapse accidental duplicate bullets such as "• • text" to one bullet.
        line = re.sub(r"^(?:[•]\s*){2,}", "• ", line)

        cleaned_lines.append(line)

    while cleaned_lines and cleaned_lines[-1] == "":
        cleaned_lines.pop()

    result = "\n".join(cleaned_lines)

    # Remove only unmatched/single Markdown asterisks while preserving valid **bold**.
    # Example: *text* -> text, while **text** remains bold in the web client.
    result = re.sub(r"(?<!\*)\*(?!\*)", "", result)

    return result


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
        answer = _clean_answer_formatting(answer)

        # Do not make temporary technical failures part of future context.
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
