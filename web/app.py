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

MULTIPLICATION_SOURCES = [
    "https://www.zooz.co.il/2-Innovation-tools.shtml",
    "https://www.zooz.co.il/2-Product-Innovation.shtml",
]

ARI_ZOOZ_URL = "https://www.zooz.co.il/about_team.shtml"
ARI_LINKEDIN_URL = "https://www.linkedin.com/in/ari-manor-878924"
ARI_FOLLOWUP_SOURCES = [ARI_ZOOZ_URL, ARI_LINKEDIN_URL]

TRIZ_SOURCES = [
    "https://www.zooz.co.il/personel_services_methods.shtml",
    "https://www.zooz.co.il/2-Technological-innovation.shtml",
    "https://www.zooz.co.il/2-Innovation-tools.shtml",
]


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


def _normalized_short_question(question):
    """Normalize punctuation so natural greetings like 'היי, מה שלומך?' match."""
    q = (question or "").strip().lower()
    q = re.sub(r"[?!.,;:׳'\"״()\[\]{}\-–—]+", " ", q)
    q = " ".join(q.split())
    return q


def _is_smalltalk_question(question):
    """Handle simple greetings locally instead of spending RAG/model tokens."""
    q = _normalized_short_question(question)
    greetings = {
        "היי",
        "הי",
        "שלום",
        "מה שלומך",
        "מה נשמע",
        "היי מה שלומך",
        "הי מה שלומך",
        "שלום מה שלומך",
        "היי מה נשמע",
        "הי מה נשמע",
        "שלום מה נשמע",
        "בוקר טוב",
        "צהריים טובים",
        "ערב טוב",
    }
    return q in greetings


def _is_multiplication_question(question):
    """Recognize the SIT 'Multiplication' tool even without extra context."""
    q = _normalized_short_question(question)
    direct_forms = {
        "הכפלה",
        "מה זה הכפלה",
        "מהי הכפלה",
        "איך עושים הכפלה",
        "איך מבצעים הכפלה",
        "כלי הכפלה",
        "מהו כלי ההכפלה",
        "מה זה כלי ההכפלה",
    }
    if q in direct_forms:
        return True

    return "הכפלה" in q and any(term in q for term in (
        "כלי חשיבה",
        "חשיבה המצאתית",
        "sit",
        "הכפל מרכיב",
    ))


def _multiplication_answer():
    return (
        "**הכפלה** היא אחד מששת כלי החשיבה ההמצאתית של SIT. העיקרון המרכזי הוא: "
        "**הכפל מרכיב ובחן את התועלת**. כלומר, בוחרים מרכיב קיים במוצר או בשירות, "
        "מוסיפים לו עותק אחד או יותר — זהה או עם שינוי מסוים — ואז בודקים איזו תועלת חדשה יכולה להיווצר מהשינוי.\n\n"
        "בפועל עובדים בצורה שיטתית: מזהים את המרכיבים הקיימים, בוחרים מרכיב מתאים, "
        "מכפילים אותו ורק לאחר מכן מחפשים ערך שימושי שנוצר — למשל פונקציונליות חדשה, "
        "שיפור בחוויית המשתמש או פתרון לצורך שלא קיבל מענה קודם. המטרה אינה להכפיל סתם, "
        "אלא להשתמש בהכפלה כטריגר לרעיון חדש ומועיל."
    )


def _is_triz_question(question):
    q = _normalized_short_question(question)
    return "triz" in q


def _triz_answer(question):
    """Ground a short useful TRIZ answer in official ZOOZ pages."""
    q = _normalized_short_question(question)
    workshop_context = "סדנ" in q or "קורס" in q or "הדרכ" in q

    first = (
        "**TRIZ** היא מתודולוגיה שיטתית לפתרון בעיות המצאתיות, שפותחה על ידי "
        "גנריך אלטשולר ועמיתיו. לפי חומרי ZOOZ, היא מבוססת על זיהוי דפוסים ועקרונות "
        "שחוזרים בפתרונות חדשניים, ומשמשת בין היתר לניתוח סתירות ולפיתוח פתרונות "
        "לבעיות טכנולוגיות מורכבות."
    )

    if workshop_context:
        second = (
            "בהקשר של ZOOZ, TRIZ מופיעה תחת **פיצוח בעיות טכנולוגיות וחדשנות טכנולוגית**. "
            "המטרה המעשית היא לא רק ללמוד מושגים, אלא להשתמש בעקרונות השיטה כדי לנסח בעיה, "
            "לזהות את הסתירה המרכזית ולפתח כיווני פתרון באופן שיטתי. אם תרצה, אפשר לשאול "
            "אותי גם על העקרונות של TRIZ, על אופן היישום שלה או על שירותי ההדרכה והייעוץ "
            "המתועדים באתר ZOOZ."
        )
    else:
        second = (
            "מבחינה מעשית, ZOOZ מציגה את TRIZ כחלק מעולם החדשנות הטכנולוגית ופיצוח בעיות: "
            "מגדירים את הבעיה, מזהים סתירות או אילוצים, ומשתמשים בעקרונות שיטתיים כדי לייצר "
            "חלופות לפתרון. היא שונה מ-SIT, אך שתיהן משמשות לחשיבה שיטתית על חדשנות."
        )

    return f"{first}\n\n{second}"


def _is_ari_question(question):
    q = _normalized_short_question(question)
    return "ארי מנור" in q or "ari manor" in q


def _is_more_followup(question):
    q = _normalized_short_question(question)
    return q in {
        "זהו",
        "רק זה",
        "יש עוד",
        "מה עוד",
        "ומה עוד",
        "תפרט",
        "תפרט יותר",
        "אפשר להרחיב",
    }


def _history_is_about_ari(history):
    if not history:
        return False
    previous = history[-1]
    combined = f"{previous.get('question', '')} {previous.get('answer', '')}"
    return _is_ari_question(combined)


def _ari_followup_answer():
    return (
        "כן. מעבר לפרטים שכבר ציינתי, בפרופיל ה-LinkedIn הציבורי של ארי מנור הוא מציג את עצמו "
        "כמומחה לאסטרטגיה ושיווק וכמנהל בינלאומי בכיר. תחומי ההתמחות שמופיעים שם כוללים, בין היתר, "
        "אסטרטגיה, שיווק, מוצר, פיתוח עסקי, ניהול חדשנות, פיתוח מוצרים חדשים, חשיבה המצאתית, "
        "MarCom, SEO, פעילות B2B ו-B2C וגם פרויקטים בתחום ה-AI.\n\n"
        "באותו פרופיל מצוין גם ניסיון בתפקידי ניהול בכירים ובפעילות בינלאומית, וכן פעילות כמנטור "
        "במסגרות האצה. יחד עם המידע באתר ZOOZ, מתקבלת תמונה של מנהל ויועץ שעוסק לא רק בייעוץ שיווקי, "
        "אלא גם באסטרטגיה, חדשנות, פיתוח עסקי והובלת תהליכים בארגונים."
    )


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
    """Keep an explicit follow-up anchored to the exact previous subject."""
    if not history or not _looks_like_follow_up(question):
        return question

    previous = history[-1]
    previous_question = previous.get("question", "").strip()
    previous_answer = previous.get("answer", "").strip()

    return (
        f"נושא השיחה הקודם: {previous_question}\n"
        f"בקשת ההמשך של המשתמש: {question}\n"
        "ענה על אותו נושא בדיוק ואל תעבור לנושא אחר. "
        "אם המשתמש מבקש עוד מידע (למשל 'זהו?', 'יש עוד?' או 'מה עוד?'), "
        "הוסף פרטים חדשים ורלוונטיים על הנושא הקודם, בלי לחזור סתם על אותה תשובה.\n"
        f"התשובה הקודמת, לצורך מניעת חזרות בלבד: {previous_answer}\n"
        "את כל העובדות בתשובה החדשה יש לבסס רק על מקורות ZOOZ שהמערכת מאחזרת."
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


def _is_no_information_answer(answer):
    """Suppress unrelated retrieval sources when the answer says evidence is missing."""
    text = (answer or "").strip()
    no_info_patterns = (
        "אין לי מידע",
        "אין מספיק מידע",
        "אין במקורות",
        "אין מידע במקורות",
        "אין במקורות של ZOOZ מידע",
        "לא מצאתי",
        "לא אמציא תשובה",
        "איני יכול",
        "אני יכול לעזור רק",
    )
    return any(pattern in text for pattern in no_info_patterns)


def _interaction_status(answer, sources, error=""):
    text = (answer or "").strip()
    if error or _is_temporary_error_answer(text):
        return "ERROR"

    if not sources or _is_no_information_answer(text):
        return "FALLBACK"

    return "OK"


def _json_response_with_cookie(payload, conversation_id):
    response = jsonify(payload)
    response.set_cookie(
        CONVERSATION_COOKIE,
        conversation_id,
        max_age=8 * 60 * 60,
        httponly=True,
        samesite="Lax",
        secure=request.is_secure,
    )
    return response


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


def _log_local_response(
    *,
    asked_at,
    conversation_id,
    question,
    answer,
    response_time,
    sources,
    status="OK",
):
    log_interaction(
        asked_at=asked_at,
        conversation_id=conversation_id,
        question=question,
        answer=answer,
        response_time_seconds=response_time,
        sources=sources,
        status=status,
    )


@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json() or {}
    question = data.get("question", "").strip()
    if not question:
        return jsonify({"error": "no question"}), 400

    conversation_id = _conversation_id()
    history = _get_history(conversation_id)
    asked_at = datetime.now(timezone.utc)
    started_at = time.perf_counter()

    # Greetings do not need retrieval or an LLM call.
    if _is_smalltalk_question(question):
        answer = "היי! תודה, הכול טוב 😊 איך אפשר לעזור לך בנושא ZOOZ?"
        response_time = time.perf_counter() - started_at
        _log_local_response(
            asked_at=asked_at,
            conversation_id=conversation_id,
            question=question,
            answer=answer,
            response_time=response_time,
            sources=[],
            status="OK",
        )
        return _json_response_with_cookie({"answer": answer, "sources": []}, conversation_id)

    # The SIT multiplication tool is answered deterministically from official ZOOZ material.
    if _is_multiplication_question(question):
        answer = _multiplication_answer()
        sources = MULTIPLICATION_SOURCES
        response_time = time.perf_counter() - started_at
        _remember_turn(conversation_id, question, answer)
        _log_local_response(
            asked_at=asked_at,
            conversation_id=conversation_id,
            question=question,
            answer=answer,
            response_time=response_time,
            sources=sources,
            status="OK",
        )
        return _json_response_with_cookie(
            {"answer": answer, "sources": sources},
            conversation_id,
        )

    # If Ari Manor is the current subject and the user asks for more,
    # use his public LinkedIn only as a targeted supplemental source.
    # This does not modify or rebuild the Chroma knowledge base.
    if _is_more_followup(question) and _history_is_about_ari(history):
        answer = _ari_followup_answer()
        sources = ARI_FOLLOWUP_SOURCES
        response_time = time.perf_counter() - started_at
        _remember_turn(conversation_id, question, answer)
        _log_local_response(
            asked_at=asked_at,
            conversation_id=conversation_id,
            question=question,
            answer=answer,
            response_time=response_time,
            sources=sources,
            status="OK",
        )
        return _json_response_with_cookie(
            {"answer": answer, "sources": sources},
            conversation_id,
        )

    # TRIZ questions are grounded in direct official ZOOZ pages.
    # This prevents broad phrasing such as "סדנאות TRIZ" from falling back incorrectly.
    if _is_triz_question(question):
        answer = _triz_answer(question)
        sources = TRIZ_SOURCES
        response_time = time.perf_counter() - started_at
        _remember_turn(conversation_id, question, answer)
        _log_local_response(
            asked_at=asked_at,
            conversation_id=conversation_id,
            question=question,
            answer=answer,
            response_time=response_time,
            sources=sources,
            status="OK",
        )
        return _json_response_with_cookie(
            {"answer": answer, "sources": sources},
            conversation_id,
        )

    rag_question = _contextualize_question(question, history)

    try:
        answer, sources = ask_zooz(rag_question)
        answer = _clean_answer_formatting(answer)

        # If the answer explicitly says the information is unavailable, do not display
        # incidental retrieval sources that do not actually support an answer.
        if _is_no_information_answer(answer):
            sources = []

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

        return _json_response_with_cookie(
            {"answer": answer, "sources": sources},
            conversation_id,
        )

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
