import sys
import os
import re
import secrets
import time
from collections import OrderedDict
from datetime import datetime, timezone
from uuid import uuid4

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, request, jsonify, render_template, Response
from dotenv import load_dotenv
from scripts.rag_engine import ask_zooz
from web.logging_store import log_interaction, export_logs_csv

load_dotenv()

app = Flask(__name__)

CONVERSATION_COOKIE = "zooz_conversation"
MAX_CONVERSATIONS = 200
MAX_TURNS = 1
MAX_ANSWER_CONTEXT_CHARS = 700
_conversations = OrderedDict()

TEMPORARY_ERROR_PREFIXES = (
    "אירעה שגיאה טכנית",
    "אירעה שגיאה זמנית",
    "השירות עמוס זמנית",
)


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
    """Contextualize only follow-ups that clearly depend on the previous turn."""
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

    exact_follow_ups = {
        "למה?", "למה", "איך?", "איך", "ומה עוד?", "ומה עוד", "מה עוד?", "מה עוד",
        "תפרט", "תפרט יותר", "אפשר להרחיב?", "אפשר להרחיב",
        "זהו?", "זהו", "רק זה?", "רק זה", "יש עוד?", "יש עוד",
    }
    return q in exact_follow_ups


def _contextualize_question(question, history):
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
        "אם שאלת ההמשך מבקשת עוד מידע (למשל 'זהו?' או 'מה עוד?'), הרחב עם פרטים נוספים ורלוונטיים. "
        "את התשובה עצמה יש לבסס רק על מקורות ZOOZ שהמערכת מאחזרת."
    )


def _clean_answer_formatting(answer):
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

        table_core = line.strip("|").strip()
        if "|" in line and table_core and re.fullmatch(r"[:\-\s|]+", table_core):
            continue

        if "|" in line:
            cells = [cell.strip() for cell in line.strip("|").split("|") if cell.strip()]
            if len(cells) >= 2:
                if any(label in cells[0] for label in ("תחום", "נושא", "קטגוריה")):
                    cleaned_lines.append("**השירותים המתאימים:**")
                else:
                    normalized_cells = [
                        re.sub(r"^(?:[•\-]\s*)+", "", cell).strip()
                        for cell in cells
                    ]
                    cleaned_lines.append("• " + " — ".join(normalized_cells))
                continue

        line = line.strip("|").strip()
        line = re.sub(r"^(?:[•]\s*){2,}", "• ", line)
        cleaned_lines.append(line)

    while cleaned_lines and cleaned_lines[-1] == "":
        cleaned_lines.pop()

    result = "\n".join(cleaned_lines)
    result = re.sub(r"(?<!\*)\*(?!\*)", "", result)
    return result


def _is_temporary_error_answer(answer):
    text = (answer or "").strip()
    return any(text.startswith(prefix) for prefix in TEMPORARY_ERROR_PREFIXES)


def _interaction_status(answer, sources, error=""):
    text = (answer or "").strip()
    if error or _is_temporary_error_answer(text):
        return "ERROR"

    fallback_phrases = (
        "אין לי מידע",
        "אין מספיק מידע",
        "לא מצאתי",
        "איני יכול",
        "אני יכול לעזור רק",
        "אין במקורות",
    )
    if not sources or any(phrase in text for phrase in fallback_phrases):
        return "FALLBACK"

    return "OK"


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/admin/logs.csv")
def download_logs():
    expected_token = os.environ.get("LOG_EXPORT_TOKEN", "").strip()
    provided_token = request.args.get("token", "").strip()

    if not expected_token or not provided_token or not secrets.compare_digest(
        expected_token, provided_token
    ):
        return jsonify({"error": "forbidden"}), 403

    try:
        csv_bytes = export_logs_csv()
        filename = f"zooz_chat_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        return Response(
            csv_bytes,
            mimetype="text/csv; charset=utf-8",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"',
                "Cache-Control": "no-store",
            },
        )
    except Exception as exc:
        return jsonify({"error": "log export failed", "details": str(exc)}), 500


@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json() or {}
    question = data.get("question", "").strip()
    if not question:
        return jsonify({"error": "no question"}), 400

    conversation_id = _conversation_id()
    history = _get_history(conversation_id)
    rag_question = _contextualize_question(question, history)

    asked_at = datetime.now(timezone.utc)
    started_at = time.perf_counter()

    try:
        answer, sources = ask_zooz(rag_question)
        answer = _clean_answer_formatting(answer)
        response_time = time.perf_counter() - started_at
        status = _interaction_status(answer, sources)

        if not _is_temporary_error_answer(answer):
            _remember_turn(conversation_id, question, answer)

        log_interaction(
            asked_at=asked_at,
            conversation_id=conversation_id,
            question=question,
            answer=answer,
            response_time_seconds=response_time,
            sources=sources,
            status=status,
        )

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
        response_time = time.perf_counter() - started_at
        log_interaction(
            asked_at=asked_at,
            conversation_id=conversation_id,
            question=question,
            answer="",
            response_time_seconds=response_time,
            sources=[],
            status="ERROR",
            error=str(exc),
        )
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)
