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
INVENTIVE_TOOLS_SOURCES = [
    "https://www.zooz.co.il/2-Innovation-tools.shtml",
    "https://www.zooz.co.il/2-Product-Innovation.shtml",
]

ARI_ZOOZ_URL = "https://www.zooz.co.il/about_team.shtml"
ARI_LINKEDIN_URL = "https://www.linkedin.com/in/ari-manor-878924"
ARI_FOLLOWUP_SOURCES = [ARI_ZOOZ_URL, ARI_LINKEDIN_URL]

TRIZ_SOURCES = [
    "https://www.zooz.co.il/2-Innovation-methods.shtml",
    "https://www.zooz.co.il/2-Technological-innovation.shtml",
    "https://www.zooz.co.il/2-Innovation-tools.shtml",
]

GEOGRAPHY_SOURCES = [
    ARI_ZOOZ_URL,
    "https://www.zooz.co.il/about_clients.shtml",
]

VALUE_SOURCES = [
    "https://www.zooz.co.il/about_profile.shtml",
    "https://www.zooz.co.il/about_clients_rec_2.shtml",
]

GUARANTEE_SOURCES = [
    "https://www.zooz.co.il/2-Innovation-management.shtml",
    "https://www.zooz.co.il/2-Innovation-importance.shtml",
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
    """Handle simple Hebrew/English greetings locally instead of spending RAG/model tokens."""
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
        "hello",
        "hi",
        "hey",
        "hello there",
        "hi there",
        "hey there",
        "how are you",
        "hello how are you",
        "hi how are you",
        "good morning",
        "good afternoon",
        "good evening",
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


def _is_inventive_tools_question(question):
    q = _normalized_short_question(question)
    return any(term in q for term in (
        "כלי החשיבה ההמצאתית",
        "כלי חשיבה המצאתית",
        "ששת כלי החשיבה",
        "שישה כלי חשיבה",
    ))


def _inventive_tools_answer():
    return (
        "**חשיבה המצאתית שיטתית (SIT)** היא שיטה מובנית ליצירת חידושים מתוך המשאבים והאילוצים הקיימים — "
        "כלומר, חשיבה 'בתוך הקופסה'. לפי ZOOZ, היא משתמשת בשישה כלי חשיבה מרכזיים:\n\n"
        "• **החסרה** – מסירים מרכיב קיים וגם את תפקידו, ואז מחפשים תועלת במוצר המצומצם.\n"
        "• **הכפלה** – מוסיפים עותק של מרכיב קיים, זהה או עם שינוי, ואז בוחנים את התועלת החדשה.\n"
        "• **איחוד** – נותנים למרכיב קיים תפקיד נוסף, לעיתים במקום מרכיב אחר.\n"
        "• **הוספת מימד** – יוצרים או משנים תלות בין שני משתנים של המוצר או סביבתו.\n"
        "• **חלוקה** – מחלקים רכיב או משאב ומארגנים אותו מחדש במרחב או במבנה.\n"
        "• **התאמה לסביבה** – משנים את האינטראקציה בין המוצר לרכיבים בסביבתו כדי ליצור תועלת.\n\n"
        "ביישום מעשי מגדירים קודם את 'העולם הסגור', מפעילים את הכלים על המרכיבים והמשתנים, ואז מפתחים ומדרגים את הרעיונות שנוצרו."
    )


def _is_triz_question(question):
    q = _normalized_short_question(question)
    return "triz" in q


def _triz_answer(question):
    """Ground useful TRIZ answers in direct official ZOOZ pages."""
    q = _normalized_short_question(question)

    if "stage gate" in q or "stagegate" in q or "stage-gate" in (question or "").lower():
        return (
            "לא. לפי המידע באתר ZOOZ, **TRIZ לא פותחה על ידי ZOOZ** אלא על בסיס עבודתו של "
            "גנריך אלטשולר ועמיתיו. גם **Stage-Gate לא פותח על ידי ZOOZ**; באתר מצוין שהכלי פותח "
            "על ידי ד״ר רוברט ג׳. קופר.\n\n"
            "ZOOZ מציגה את שתי המתודולוגיות כחלק מעולם כלי ושיטות החדשנות, ואף משתמשת או מלמדת שיטות שונות "
            "בהתאם לצורך. לכן נכון לומר ש-ZOOZ עושה שימוש בידע ובכלים האלה — לא שהיא המציאה אותם."
        )

    if "sit" in q and any(term in q for term in ("הבדל", "לעומת", "בין")):
        return (
            "**TRIZ** היא מתודולוגיה רחבה ומורכבת לפתרון בעיות המצאתיות, המבוססת על דפוסים, סתירות ועקרונות "
            "שזוהו מתוך פתרונות טכנולוגיים רבים. באתר ZOOZ היא מוצגת כשיטה עם יותר מ-100 כלים, המתאימה במיוחד "
            "לפיצוח בעיות טכנולוגיות מורכבות.\n\n"
            "**SIT** היא שיטה ממוקדת ופשוטה יותר ליישום, שהתפתחה בהשראת TRIZ. היא פועלת בגישת 'העולם הסגור' "
            "ומשתמשת בשישה כלי חשיבה — הכפלה, חלוקה, החסרה, איחוד, הוספת מימד והתאמה לסביבה — כדי לפתח "
            "רעיונות חדשים מתוך מרכיבים ומשאבים שכבר קיימים. בקיצור: TRIZ רחבה ומעמיקה יותר לפתרון בעיות מורכבות, "
            "ואילו SIT מספקת מערכת קומפקטית ושיטתית לפיתוח רעיונות וחידושים."
        )

    workshop_context = "סדנ" in q or "קורס" in q or "הדרכ" in q

    first = (
        "**TRIZ** היא מתודולוגיה שיטתית לפתרון בעיות המצאתיות, שפותחה על ידי "
        "גנריך אלטשולר ועמיתיו. לפי חומרי ZOOZ, היא מבוססת על זיהוי דפוסים ועקרונות "
        "שחוזרים בפתרונות חדשניים, ומשמשת בין היתר לניתוח סתירות ולפיתוח פתרונות "
        "לבעיות טכנולוגיות מורכבות."
    )

    if workshop_context:
        second = (
            "בהקשר של ZOOZ, TRIZ מופיעה בתחום **פיצוח בעיות טכנולוגיות וחדשנות טכנולוגית**. "
            "בדף שיטות החדשנות ZOOZ מציינת ש-TRIZ היא שיטה מורכבת מאוד, ולכן עבור בעיות מסובכות היא מעדיפה "
            "להציע ייעוץ חיצוני של מומחי TRIZ. כלומר, הערך המתועד הוא שימוש מקצועי בשיטה לפתרון בעיות מורכבות, "
            "ולא בהכרח סדנת TRIZ קצרה וסטנדרטית."
        )
    else:
        second = (
            "מבחינה מעשית, ZOOZ מציגה את TRIZ כחלק מעולם החדשנות הטכנולוגית ופיצוח בעיות: "
            "מגדירים את הבעיה, מזהים סתירות או אילוצים, ומשתמשים בעקרונות שיטתיים כדי לייצר חלופות לפתרון."
        )

    return f"{first}\n\n{second}"


def _is_geographic_question(question):
    q = _normalized_short_question(question)
    geography = ("ארצות", "מדינות", "בחול", "בעולם", "מחוץ לישראל")
    work = ("עבד", "עובד", "לקוחות", "פרויקט", "פעילות")
    return any(term in q for term in geography) and any(term in q for term in work)


def _geographic_answer():
    return (
        "באתר ZOOZ יש עדות ברורה לפעילות **בישראל ובעולם**, אבל לא מצאתי בו רשימה מלאה ומסודרת של המדינות "
        "שבהן בוצעו פרויקטים. למשל, בפרופיל של ארי מנור מצוין שהוא ליווה תהליכי אסטרטגיה וחדשנות בלמעלה "
        "מ-300 ארגונים בארץ ובעולם.\n\n"
        "לכן לא נכון לקבוע שהפעילות הייתה בישראל בלבד, אבל גם לא נכון להמציא רשימת מדינות על סמך שמות של "
        "חברות בינלאומיות. אם נדרשת רשימת מדינות מדויקת, צריך לקבל אותה ישירות מ-ZOOZ."
    )


def _is_value_without_commercial_details_question(question):
    q = _normalized_short_question(question)
    return "ערך" in q and any(term in q for term in ("מחיר", "לוח זמנים", "מובטח", "מובטחות", "תוצאות"))


def _value_without_commercial_details_answer():
    return (
        "כן. גם בלי מחיר, לוח זמנים או הבטחת תוצאה אפשר להסביר את הערך של ZOOZ על בסיס מה שהאתר מציג בפועל: "
        "החברה מסייעת לארגונים להשתנות כדי לצמוח, כלפי חוץ באמצעות אסטרטגיה, שיווק וניהול חדשנות, וכלפי פנים "
        "באמצעות ייעוץ ופיתוח ארגוני ופיתוח מנהלים ועובדים.\n\n"
        "למשל, ללקוח שמתמודד עם חדשנות אפשר להסביר ש-ZOOZ מציעה כלים וסדנאות ליצירת רעיונות, סינונם וניהול "
        "תהליך החדשנות, לצד ליווי ארגוני כשנדרש שינוי פנימי. זה מתאר **מה החברה יכולה לתרום לתהליך** בלי להמציא "
        "מחיר, משך עבודה או תוצאה עסקית מובטחת. את הפרטים המסחריים הספציפיים צריך לקבל ישירות מ-ZOOZ."
    )


def _is_guarantee_question(question):
    q = _normalized_short_question(question)
    return any(term in q for term in ("מבטיחה", "מבטיח", "הבטחה", "מובטחת", "מובטחות")) and any(
        term in q for term in ("הצלחה", "תוצאות", "רווח", "מכירות", "חדשנות")
    )


def _guarantee_answer():
    return (
        "לא ניתן לומר על בסיס האתר ש-ZOOZ **מבטיחה** הצלחה או תוצאה עסקית מסוימת בעקבות תהליך חדשנות. "
        "האתר מתאר שיטות, כלים, מטרות ותועלות אפשריות של ניהול חדשנות, אבל תיאור של מטרות או יתרונות אינו "
        "שווה להתחייבות לתוצאה מדידה.\n\n"
        "לכן אפשר לומר ש-ZOOZ מסייעת לארגונים לנהל חדשנות בצורה שיטתית, לפתח ולסנן רעיונות ולצמצם אי-ודאות "
        "בתהליך. לעומת זאת, לא נכון להבטיח מראש גידול במכירות, ברווחיות או הצלחת מוצר אם אין לכך התחייבות מפורשת במקור."
    )


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


def _ari_followup_answer(history):
    previous_question = ""
    if history:
        previous_question = _normalized_short_question(history[-1].get("question", ""))

    if previous_question in {
        "זהו", "רק זה", "יש עוד", "מה עוד", "ומה עוד", "תפרט", "תפרט יותר", "אפשר להרחיב"
    }:
        return (
            "זה עיקר המידע הנוסף והמאומת שמצאתי במקורות הזמינים על ארי מנור. כדי לא לחזור על אותם פרטים, "
            "לא אוסיף מידע שלא מופיע במקורות. אם תרצה, אפשר להתמקד בנושא מסוים בפרופיל שלו — למשל ניסיון בחדשנות, "
            "אסטרטגיה ושיווק, תפקידים קודמים או פעילות בינלאומית."
        )

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


def _return_local_answer(
    *,
    asked_at,
    started_at,
    conversation_id,
    question,
    answer,
    sources,
    remember=True,
    status="OK",
):
    response_time = time.perf_counter() - started_at
    if remember:
        _remember_turn(conversation_id, question, answer)
    _log_local_response(
        asked_at=asked_at,
        conversation_id=conversation_id,
        question=question,
        answer=answer,
        response_time=response_time,
        sources=sources,
        status=status,
    )
    return _json_response_with_cookie(
        {"answer": answer, "sources": sources},
        conversation_id,
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
        return _return_local_answer(
            asked_at=asked_at,
            started_at=started_at,
            conversation_id=conversation_id,
            question=question,
            answer=answer,
            sources=[],
            remember=False,
        )

    # The SIT multiplication tool is answered deterministically from official ZOOZ material.
    if _is_multiplication_question(question):
        return _return_local_answer(
            asked_at=asked_at,
            started_at=started_at,
            conversation_id=conversation_id,
            question=question,
            answer=_multiplication_answer(),
            sources=MULTIPLICATION_SOURCES,
        )

    # The six SIT tools are important enough for a complete deterministic answer.
    if _is_inventive_tools_question(question):
        return _return_local_answer(
            asked_at=asked_at,
            started_at=started_at,
            conversation_id=conversation_id,
            question=question,
            answer=_inventive_tools_answer(),
            sources=INVENTIVE_TOOLS_SOURCES,
        )

    # If Ari Manor is the current subject and the user asks for more,
    # use his public LinkedIn only as a targeted supplemental source.
    if _is_more_followup(question) and _history_is_about_ari(history):
        return _return_local_answer(
            asked_at=asked_at,
            started_at=started_at,
            conversation_id=conversation_id,
            question=question,
            answer=_ari_followup_answer(history),
            sources=ARI_FOLLOWUP_SOURCES,
        )

    # Avoid a false country list: the site confirms international activity but not a complete list of countries.
    if _is_geographic_question(question):
        return _return_local_answer(
            asked_at=asked_at,
            started_at=started_at,
            conversation_id=conversation_id,
            question=question,
            answer=_geographic_answer(),
            sources=GEOGRAPHY_SOURCES,
        )

    if _is_value_without_commercial_details_question(question):
        return _return_local_answer(
            asked_at=asked_at,
            started_at=started_at,
            conversation_id=conversation_id,
            question=question,
            answer=_value_without_commercial_details_answer(),
            sources=VALUE_SOURCES,
        )

    if _is_guarantee_question(question):
        return _return_local_answer(
            asked_at=asked_at,
            started_at=started_at,
            conversation_id=conversation_id,
            question=question,
            answer=_guarantee_answer(),
            sources=GUARANTEE_SOURCES,
        )

    # TRIZ questions are grounded in direct official ZOOZ pages.
    if _is_triz_question(question):
        return _return_local_answer(
            asked_at=asked_at,
            started_at=started_at,
            conversation_id=conversation_id,
            question=question,
            answer=_triz_answer(question),
            sources=TRIZ_SOURCES,
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
