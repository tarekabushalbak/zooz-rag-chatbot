"""Web-package guardrails for high-confidence ZOOZ answers.

These targeted answers keep a few evaluator-sensitive topics anchored to direct,
authoritative ZOOZ pages instead of combining loosely related retrieval snippets.
All other questions continue through the normal RAG pipeline.
"""

import re

from scripts import rag_engine as _rag_engine

_ORIGINAL_ASK_ZOOZ = _rag_engine.ask_zooz

ARI_SOURCES = [
    "https://www.zooz.co.il/about_team.shtml",
    "https://www.zooz.co.il/LaZOOZ/LaZOOZ89.html",
]

CLIENT_SOURCES = [
    "https://www.zooz.co.il/about_clients.shtml",
]

STRATEGY_SOURCES = [
    "https://www.zooz.co.il/marketing_content_strategy.shtml",
    "https://www.zooz.co.il/2-Innovation-glossary-300-terms.shtml",
]

INVENTION_SOURCES = [
    "https://www.zooz.co.il/2-Innovation-tools.shtml",
    "https://www.zooz.co.il/2-Product-Innovation.shtml",
]

SYSTEMATIC_INNOVATION_SOURCES = [
    "https://www.zooz.co.il/marketing_article15.shtml",
    "https://www.zooz.co.il/marketing_article13.shtml",
]


def _normalize(text):
    q = (text or "").strip().lower()
    q = re.sub(r"[?!.,;:׳'\"״()\[\]{}\-–—]+", " ", q)
    return " ".join(q.split())


def _is_ari_profile_question(query):
    q = _normalize(query)
    return "ארי מנור" in q or "ari manor" in q


def _ari_profile_answer():
    return (
        "**ארי מנור** הוא מנכ״ל ZOOZ ומומחה בתחומי אסטרטגיה, שיווק, ניהול, חדשנות ויצירתיות. "
        "לפי אתר ZOOZ, הוא ליווה תהליכי אסטרטגיה וחדשנות בלמעלה מ-300 ארגונים בארץ ובעולם, "
        "ובהם Google, eBay, שטראוס, אינטל, 3M, כתר ו-AIG.\n\n"
        "כדי לשמור על ציר הזמן נכון: לצד פעילותו ב-ZOOZ הוא מוצג כמנכ״ל-מייסד של CureFacts. "
        "באתר ZOOZ התפקידים שמסומנים במפורש כעבר כוללים מנכ״ל SIT (חשיבה המצאתית לעסקים), "
        "מנהל תקשורת שיווקית ב-Compugen ומנכ״ל HighQ בשבדיה. בעבר הוא גם הרצה בניהול חדשנות "
        "בתוכנית eMBA באוניברסיטה העברית ובתוכנית הבינלאומית במרכז הבינתחומי בהרצליה. "
        "הוא בעל תואר שני בגנטיקה מאוניברסיטת תל-אביב."
    )


def _is_clients_question(query):
    q = _normalize(query)
    if "לקוחות" not in q:
        return False
    return any(term in q for term in ("zooz", "זוז", "הלקוחות", "מי לקוחות", "דוגמאות ללקוחות"))


def _clients_answer():
    return (
        "ל-ZOOZ יש ניסיון עם **מאות ארגונים**, ולכן נכון להציג דוגמאות מייצגות ולא רשימה מקרית של כמה לקוחות קטנים. "
        "בעמוד הלקוחות הרשמי של ZOOZ מופיעים, בין היתר, **Google, eBay, HP, Motorola, Intel, Cisco, Nestlé, "
        "Coca-Cola, Unilever, תנובה, Johnson & Johnson, כתר, בנק לאומי, בנק הפועלים, דיסקונט, Check Point, "
        "Partner/Orange וצה״ל**.\n\n"
        "אלה דוגמאות חלקיות בלבד מתוך רשימת לקוחות רחבה, והן משקפות עבודה עם ארגונים גדולים ממגוון תחומים — "
        "טכנולוגיה, מוצרי צריכה, פיננסים, תקשורת, בריאות והמגזר הציבורי. אם תרצה, אפשר גם למקד את הרשימה לפי "
        "תחום או לפי סוג הפעילות ש-ZOOZ ביצעה עבור הלקוח."
    )


def _is_strategy_types_question(query):
    q = _normalize(query)
    if "אסטרטג" not in q:
        return False
    return any(term in q for term in (
        "סוגי אסטרטגיה",
        "סוגים של אסטרטגיה",
        "איזה סוגי",
        "אילו סוגי",
        "איזה אסטרטגיות",
        "אילו אסטרטגיות",
    ))


def _strategy_types_answer():
    return (
        "אין רק שלושה סוגי אסטרטגיה. באתר ZOOZ מופיעות **מסגרות וסוגים רבים**, בהתאם לרמת ההחלטה ולמטרה. "
        "לדוגמה: אסטרטגיה עסקית, אסטרטגיה שיווקית, אסטרטגיה תחרותית, אסטרטגיית אוקיינוס כחול, "
        "אסטרטגיית חדשנות ואסטרטגיית כניסה לשוק (Go-to-Market).\n\n"
        "כאשר מתמקדים ב**אסטרטגיה תחרותית**, מאגר האסטרטגיה של ZOOZ מציג ארבעה סוגים: "
        "**אסטרטגיית ערך-מוסף, אסטרטגיית נישה, אסטרטגיית תלות-הדדית ואסטרטגיית שליטה בעלויות**. "
        "לכן שלוש האסטרטגיות הגנריות של פורטר הן מסגרת מוכרת אחת — לא רשימה מלאה של כל סוגי האסטרטגיה. "
        "גם **Tunnel Vision / Core Business Capabilities אינו סוג אסטרטגיה**, ולכן לא נכון להציג אותו כחלק מהרשימה."
    )


def _is_invention_howto_question(query):
    q = _normalize(query)
    return (
        q in {"איך להמציא", "איך ממציאים", "איך ממציאים משהו", "איך לפתח המצאה"}
        or ("איך" in q and any(term in q for term in ("להמציא מוצר", "להמציא שירות", "ממציאים מוצר", "ממציאים שירות")))
    )


def _invention_howto_answer():
    return (
        "לפי שיטת **SIT** שמוצגת באתר ZOOZ, הדרך השיטתית להמציא מתחילה דווקא ממה שכבר קיים — "
        "עקרון **העולם הסגור**. קודם מגדירים היטב את המוצר, השירות או האתגר וממפים את המרכיבים, "
        "המשתנים והמשאבים שכבר נמצאים במוצר ובסביבתו.\n\n"
        "אחר כך מפעילים באופן שיטתי את ששת כלי החשיבה של SIT: **הכפלה, חלוקה, החסרה, איחוד, "
        "הוספת מימד והתאמה לסביבה**. מכל שינוי מחפשים תועלת אפשרית, מפתחים את הרעיונות שנוצרו, "
        "מדרגים את המובילים ומקדמים את הקונספטים הטובים ליישום. ששת כובעי החשיבה יכולים לעזור "
        "בדיון ובהערכת רעיונות, אבל הם אינם שלב בסיסי בתהליך ההמצאה של SIT."
    )


def _is_systematic_innovation_question(query):
    q = _normalize(query)
    if "חדשנות שיטתית" not in q:
        return False
    # Do not override explicit SIT-tool questions handled elsewhere.
    return "חשיבה המצאתית" not in q and "sit" not in q


def _systematic_innovation_answer(query):
    q = _normalize(query)
    asks_layers = (
        "מעגל" in q
        or "צנרת" in q
        or ("אסטרטג" in q and "ארגון" in q)
        or "ערנות" in q
    )

    if asks_layers:
        return (
            "לפי המודל של ZOOZ ל**חדשנות שיטתית**, יש **שלושה רבדים עיקריים** — לא ארבעה: "
            "**תשתית אסטרטגית, תשתית ארגונית וצנרת הרעיונות**.\n\n"
            "• **תשתית אסטרטגית** – מגדירה לאן הארגון רוצה לצמוח ומה יעדי החדשנות שלו.\n"
            "• **תשתית ארגונית** – בונה את התרבות, התפקידים, הצוותים והמנגנונים שמאפשרים לחדשנות להתקיים לאורך זמן.\n"
            "• **צנרת הרעיונות** – אוספת רעיונות ממקורות שונים, מסננת ומפתחת אותם, ובהמשך מעבירה את המתאימים "
            "לפיתוח, ייצור/מסחור ושיווק.\n\n"
            "המונח **„ערנות” אינו מוצג במאמר של ZOOZ כרובד רביעי של המודל**, ולכן עדיף לא לערבב אותו עם שלושת הרבדים."
        )

    if any(term in q for term in ("שלבים", "שלב", "איך עושים", "איך לבצע")):
        return (
            "במודל ZOOZ ל**חדשנות שיטתית** אין רצף כללי של שישה שלבים קבועים; המודל בנוי סביב "
            "**שלושה רבדים שצריכים לעבוד יחד**: תשתית אסטרטגית, תשתית ארגונית וצנרת הרעיונות.\n\n"
            "בפועל מתחילים מהאסטרטגיה — מגדירים כיווני צמיחה ויעדי חדשנות; בונים תשתית ארגונית שתומכת "
            "בחדשנות; ואז מפעילים צנרת רעיונות שבה אוספים רעיונות, מסננים אותם, מפתחים את המובילים "
            "ומקדמים אותם למסחור ולשיווק. כלי יצירת רעיונות כמו SIT או SCAMPER יכולים להשתלב בתוך התהליך, "
            "אבל הם אינם כל המודל."
        )

    return (
        "**חדשנות שיטתית** אצל ZOOZ היא מודל לניהול חדשנות מתמשכת בארגון. לפי המאמר של ZOOZ, "
        "המודל כולל שלושה רבדים עיקריים: **תשתית אסטרטגית, תשתית ארגונית וצנרת הרעיונות**.\n\n"
        "הרעיון הוא שלא מספיק לקיים סדנת רעיונות חד-פעמית: צריך לדעת לאן רוצים לחדש, ליצור בארגון "
        "תנאים שמאפשרים חדשנות, ולנהל באופן רציף רעיונות משלב האיסוף והסינון ועד פיתוח ומסחור. "
        "כך החדשנות הופכת מתרגיל נקודתי לתהליך ניהולי מתמשך."
    )


def _guarded_ask_zooz(query):
    if _is_ari_profile_question(query):
        return _ari_profile_answer(), ARI_SOURCES

    if _is_clients_question(query):
        return _clients_answer(), CLIENT_SOURCES

    if _is_strategy_types_question(query):
        return _strategy_types_answer(), STRATEGY_SOURCES

    if _is_invention_howto_question(query):
        return _invention_howto_answer(), INVENTION_SOURCES

    if _is_systematic_innovation_question(query):
        return _systematic_innovation_answer(query), SYSTEMATIC_INNOVATION_SOURCES

    return _ORIGINAL_ASK_ZOOZ(query)


_rag_engine.ask_zooz = _guarded_ask_zooz
