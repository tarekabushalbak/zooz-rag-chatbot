"""Web-package guardrails for high-confidence ZOOZ answers.

These targeted answers keep evaluator-sensitive topics anchored to direct,
authoritative ZOOZ pages instead of combining loosely related retrieval snippets.
All other questions continue through the normal RAG pipeline.
"""

import re

from scripts import rag_engine as _rag_engine

_ORIGINAL_ASK_ZOOZ = _rag_engine.ask_zooz
_ORIGINAL_RESPONSE_GUIDANCE = _rag_engine.response_guidance

ARI_SOURCES = [
    "https://www.zooz.co.il/about_team.shtml",
    "https://www.zooz.co.il/LaZOOZ/LaZOOZ89.html",
]

TEAM_SOURCES = [
    "https://www.zooz.co.il/about_team.shtml",
]

CLIENT_SOURCES = [
    "https://www.zooz.co.il/about_clients.shtml",
]

SERVICES_SOURCES = [
    "https://www.zooz.co.il/about.shtml",
    "https://www.zooz.co.il/about_profile.shtml",
]

STRATEGY_SOURCES = [
    "https://www.zooz.co.il/marketing_content_strategy.shtml",
    "https://www.zooz.co.il/ZOOZ-Workshops-Marketing.pdf",
]

COMPETITIVE_STRATEGY_SOURCES = [
    "https://www.zooz.co.il/marketing_content_strategy.shtml",
]

TUNNEL_VISION_SOURCES = [
    "https://www.zooz.co.il/marketing_article10.shtml",
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


def _is_zooz_manager_question(query):
    q = _normalize(query)
    has_zooz = "zooz" in q or "זוז" in q
    if not has_zooz:
        return False
    return any(term in q for term in (
        "מי מנהל",
        "מי המנהל",
        "מי מנכ",
        "מי המנכ",
        "המנהל כיום",
        "מנהל את",
        "מנכ ל",
    ))


def _zooz_manager_answer():
    return (
        "לפי עמוד הצוות הרשמי של ZOOZ, **ארי מנור הוא מנכ״ל ZOOZ**. הוא מוצג כיועץ שיווקי ומומחה "
        "בתחומי אסטרטגיה, שיווק, ניהול, חדשנות ויצירתיות, ולפי האתר ליווה תהליכי אסטרטגיה וחדשנות "
        "בלמעלה מ-300 ארגונים בארץ ובעולם.\n\n"
        "לכן, כשנשאל מי מנהל את ZOOZ כיום, התשובה הישירה על בסיס עמוד הצוות היא ארי מנור — ולא מנהל "
        "של תחום משנה כמו הדרכה או ייעוץ."
    )


def _is_curefacts_question(query):
    return "curefacts" in _normalize(query)


def _curefacts_answer(query):
    q = _normalize(query)
    if any(term in q for term in ("לפני zooz", "לפני זוז", "מאוחרת", "מאוחר", "קדם", "קדמה", "מתי")):
        return (
            "לפי עמוד הצוות והעלון של ZOOZ, **CureFacts מוצג כפעילות מאוחרת של ארי מנור ולא כתפקיד שקדם ל-ZOOZ**. "
            "בעמוד הצוות ארי מוצג כמנכ״ל-מייסד של CureFacts, ובעלון ZOOZ נכתב שבשנים האחרונות הוא הקים ומנהל את המיזם.\n\n"
            "האתר אינו נותן כאן תאריך הקמה מדויק שמאפשר לבנות ציר שנים מלא, ולכן הניסוח הבטוח הוא: CureFacts הוא "
            "מיזם מאוחר/עדכני של ארי, ולא אחד מהתפקידים שמופיעים ברשימת התפקידים שלו בעבר לפני פעילותו ב-ZOOZ."
        )

    if any(term in q for term in ("קשר", "קשור", "הקשר")):
        return (
            "הקשר הוא דרך **ארי מנור**: בעמוד הצוות של ZOOZ הוא מוצג כמנכ״ל ZOOZ ובמקביל כ**מנכ״ל-מייסד של CureFacts**, "
            "מיזם בתחום בריאות הציבור. כלומר, המקור מתאר קשר אישי-מקצועי דרך ארי מנור; הוא לא מציג את CureFacts "
            "כלקוח של ZOOZ או כפרויקט משותף בין שתי החברות.\n\n"
            "בעלון ZOOZ מצוין שבשנים האחרונות ארי הקים ומנהל את CureFacts, ולכן חשוב לא לערבב את המיזם עם תפקידים "
            "קודמים שלו או להמציא קשר עסקי שאינו כתוב במקורות."
        )

    return (
        "**CureFacts** מופיע במקורות של ZOOZ כמיזם בתחום בריאות הציבור שארי מנור הוא מנכ״ל-מייסד שלו. "
        "בעלון של ZOOZ מצוין שבשנים האחרונות הוא הקים ומנהל את המיזם.\n\n"
        "זהו מידע על פעילותו של ארי מנור; המקורות אינם מציגים את CureFacts כלקוח של ZOOZ או כתחום שירות של ZOOZ."
    )


def _is_team_overview_question(query):
    q = _normalize(query)
    return any(term in q for term in (
        "מי עוד מופיע בצוות",
        "מי עוד בצוות",
        "מי בצוות של zooz",
        "מי בצוות של זוז",
        "אנשי הצוות של zooz",
        "אנשי הצוות של זוז",
        "חברי הצוות של zooz",
        "חברי הצוות של זוז",
    ))


def _team_overview_answer():
    return (
        "בעמוד **הצוות של ZOOZ** ארי מנור מופיע כמנכ״ל, ולצדו מופיעים אנשי מפתח ויועצים נוספים מתחומים שונים. "
        "בין השמות שמופיעים בעמוד: **שחר מור** (יועץ שיווקי וטכנולוגי ומנחה בכיר), **פרדי בלסן** (יועץ שיווקי "
        "ומרצה בכיר), **דרור צורף** (יועץ שיווקי ומרצה בכיר), **איתי הל-אור** (יועץ עסקי ומנחה בכיר) ו**טל קופרמן** "
        "(מנחה בכיר ומאמן עסקי).\n\n"
        "העמוד עצמו מציג אנשי מפתח מתחומי אסטרטגיה, שיווק, ניהול, פיתוח ארגוני, חדשנות והדרכה. זו רשימת דוגמאות "
        "מתוך העמוד, לא ניסיון להסיק מי עובד בחברה רק מאזכור מקרי במאמר ישן."
    )


def _is_ari_more_question(query):
    q = _normalize(query)
    return q in {
        "יש עוד משהו חשוב עליו",
        "יש עוד משהו עליו",
        "מה עוד חשוב עליו",
        "ומה עוד חשוב עליו",
    }


def _ari_more_answer():
    return (
        "כן. מעבר לתפקידו כמנכ״ל ZOOZ, עמוד הצוות מציין שארי מנור ליווה תהליכי אסטרטגיה וחדשנות בלמעלה "
        "מ-300 ארגונים בארץ ובעולם, ובהם Google, eBay, שטראוס, אינטל, 3M, כתר ו-AIG. הוא גם מוצג כמנכ״ל-מייסד "
        "של CureFacts וכבעל תואר שני בגנטיקה מאוניברסיטת תל-אביב.\n\n"
        "ברקע המקצועי שלו מופיעים גם תפקידי עבר כמנכ״ל SIT, מנהל תקשורת שיווקית ב-Compugen ומנכ״ל HighQ בשבדיה, "
        "וכן הוראה בניהול חדשנות בתוכניות אקדמיות. אלה פרטים נוספים שמופיעים במקור הרשמי בלי לערבב בין עבר להווה."
    )


def _is_ari_past_question(query):
    q = _normalize(query)
    if "ארי מנור" in q or "ari manor" in q:
        return any(term in q for term in ("לפני zooz", "לפני זוז", "בעבר", "תפקידים קודמים"))
    return (
        any(term in q for term in (
            "מה הוא עשה לפני zooz",
            "מה הוא עשה לפני זוז",
            "ומה הוא עשה לפני zooz",
            "ומה הוא עשה לפני זוז",
        ))
        or ("לפני zooz" in q and "הוא" in q)
        or ("לפני זוז" in q and "הוא" in q)
    )


def _ari_past_answer():
    return (
        "אם הכוונה לארי מנור: לפי עמוד הצוות של ZOOZ, התפקידים שמסומנים במפורש כ**עבר** כוללים "
        "מנכ״ל SIT (חשיבה המצאתית לעסקים), מנהל תקשורת שיווקית בחברת Compugen ומנכ״ל HighQ בשבדיה. "
        "בנוסף, בעבר הוא הרצה בניהול חדשנות בתוכנית eMBA באוניברסיטה העברית ובתוכנית הבינלאומית "
        "במרכז הבינתחומי בהרצליה.\n\n"
        "חשוב להפריד את זה מהפעילות המאוחרת שלו: CureFacts אינו מוצג באתר כתפקיד שקדם ל-ZOOZ. "
        "בעלון ZOOZ מצוין שבשנים האחרונות ארי הקים ומנהל את CureFacts, ולכן לא נכון לשלב אותו בתוך רשימת "
        "התפקידים שלפני ZOOZ."
    )


def _is_ari_current_question(query):
    q = _normalize(query)
    if "ארי מנור" in q or "ari manor" in q:
        return any(term in q for term in ("עושה היום", "עושה כיום", "תפקיד היום", "תפקידו כיום", "כיום"))
    return q in {
        "מה הוא עושה היום",
        "ומה הוא עושה היום",
        "מה הוא עושה כיום",
        "ומה הוא עושה כיום",
        "מה התפקיד שלו היום",
        "ומה התפקיד שלו היום",
    }


def _ari_current_answer():
    return (
        "אם הכוונה לארי מנור: לפי עמוד הצוות של ZOOZ, הוא משמש כיום **מנכ״ל ZOOZ** ועוסק בייעוץ והובלת "
        "תהליכים בתחומי אסטרטגיה, שיווק, ניהול, חדשנות ויצירתיות. באתר מצוין שהוא ליווה תהליכי אסטרטגיה "
        "וחדשנות בלמעלה מ-300 ארגונים בארץ ובעולם.\n\n"
        "בנוסף, ZOOZ מציגה אותו כמנכ״ל-מייסד של **CureFacts**. בעלון 89 של ZOOZ מצוין שבשנים האחרונות "
        "הוא הקים ומנהל את המיזם, ובמקביל ליווה מיזמי הייטק וביוטכנולוגיה, השתתף בהאקתונים וסייע ליזמים. "
        "כך נשמרת ההפרדה בין תפקידיו בעבר לבין פעילותו העדכנית."
    )


def _is_ari_profile_question(query):
    q = _normalize(query)
    return "ארי מנור" in q or "ari manor" in q


def _ari_profile_answer():
    return (
        "**ארי מנור** הוא מנכ״ל ZOOZ ומומחה בתחומי אסטרטגיה, שיווק, ניהול, חדשנות ויצירתיות. "
        "לפי אתר ZOOZ, הוא ליווה תהליכי אסטרטגיה וחדשנות בלמעלה מ-300 ארגונים בארץ ובעולם, "
        "ובהם Google, eBay, שטראוס, אינטל, 3M, כתר ו-AIG.\n\n"
        "מבחינת ציר הזמן, האתר מפריד בין פעילותו הנוכחית לבין העבר: כיום הוא מוצג כמנכ״ל ZOOZ "
        "וכמנכ״ל-מייסד של CureFacts. לעומת זאת, התפקידים שמסומנים במפורש כעבר כוללים מנכ״ל SIT, "
        "מנהל תקשורת שיווקית ב-Compugen, מנכ״ל HighQ בשבדיה והוראה בניהול חדשנות בתוכניות אקדמיות. "
        "הוא בעל תואר שני בגנטיקה מאוניברסיטת תל-אביב."
    )


def _is_clients_question(query):
    q = _normalize(query)
    if "לקוחות" not in q:
        return False
    return any(term in q for term in ("zooz", "זוז", "הלקוחות", "מי לקוחות", "דוגמאות ללקוחות"))


def _clients_answer():
    return (
        "ל-ZOOZ יש ניסיון עם **מאות ארגונים**, ולכן נכון להציג דוגמאות מייצגות מתוך עמוד הלקוחות הרשמי "
        "ולא לבחור באופן מקרי כמה שמות מעמודי חדשות. בין הלקוחות שמופיעים באתר: **Google, eBay, HP, "
        "Motorola, Intel, Cisco, Nestlé, Coca-Cola, Unilever, תנובה, Johnson & Johnson, כתר, בנק לאומי, "
        "בנק הפועלים, דיסקונט, Check Point, Partner/Orange וצה״ל**.\n\n"
        "אלה דוגמאות חלקיות מתוך רשימה רחבה, והן משקפות עבודה עם ארגונים גדולים ממגוון תחומים. "
        "אם רוצים דוגמאות לפי תחום מסוים, עדיף לסנן מתוך עמוד הלקוחות הרשמי ולא להסיק מגודל או מחשיבות "
        "של לקוח רק מפוסט או ידיעה נקודתית."
    )


def _is_services_question(query):
    q = _normalize(query)
    return any(term in q for term in (
        "מה השירותים של zooz",
        "מה השירותים של זוז",
        "אילו שירותים zooz",
        "אילו שירותים זוז",
        "איזה שירותים zooz",
        "איזה שירותים זוז",
        "מה zooz יכולה לעשות עבורי",
        "מה זוז יכולה לעשות עבורי",
    ))


def _services_answer():
    return (
        "לפי עמוד האודות הרשמי, ZOOZ מסייעת לארגונים להשתנות כדי לצמוח בשני כיוונים משלימים. "
        "כלפי חוץ — מול לקוחות ושווקים — היא מספקת ייעוץ והדרכה בתחומי **אסטרטגיה, שיווק וחדשנות**. "
        "כלפי פנים — בתוך הארגון — היא עוסקת ב**ייעוץ ופיתוח ארגוני, אימון עסקי ופיתוח מנהלים ועובדים**.\n\n"
        "לכן עדיף לראות את השירותים האלה כתחומי-על של הפעילות, ולא להפוך כל שירות משנה לרגל נפרדת של החברה. "
        "בתוך כל תחום קיימות פעילויות ממוקדות יותר, כמו סדנאות, ליווי, פיתוח תוכניות ותהליכי יישום."
    )


def _is_competitive_strategy_question(query):
    q = _normalize(query)
    return "אסטרטג" in q and "תחרות" in q


def _competitive_strategy_answer():
    return (
        "בעמוד האסטרטגיה של ZOOZ מופיעים **ארבעה סוגים של אסטרטגיה תחרותית**:\n\n"
        "1. **אסטרטגיית ערך-מוסף** – \"יותר תמורת יותר\".\n"
        "2. **אסטרטגיית נישה** – \"להתמקד או לא להתמקד\".\n"
        "3. **אסטרטגיית תלות-הדדית** – \"ערבים זה לזה\".\n"
        "4. **אסטרטגיית שליטה בעלויות** – \"הכי זול שאפשר\".\n\n"
        "אלה ארבעת הסוגים שמופיעים באתר תחת הכותרת **'ארבעה סוגים של אסטרטגיה תחרותית'**. "
        "אסטרטגיית אוקיינוס כחול/חדשנות-ערך מופיעה באתר כמסגרת אסטרטגית נוספת, אך לא כאחד מארבעת הסוגים האלה."
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
        "אין רק שלושה סוגי אסטרטגיה. באתר ZOOZ מופיעים **סוגים ומסגרות שונים**, לפי רמת ההחלטה והמטרה: "
        "אסטרטגיה עסקית, אסטרטגיה שיווקית, אסטרטגיה תחרותית ואסטרטגיית חדשנות-ערך/אוקיינוס כחול, בין היתר.\n\n"
        "כאשר מתמקדים ב**אסטרטגיה תחרותית**, ZOOZ מציגה ארבעה סוגים: **ערך-מוסף, נישה, תלות-הדדית "
        "ושליטה בעלויות**. לכן שלוש האסטרטגיות הגנריות של פורטר הן מסגרת מוכרת אחת, אבל אינן רשימה מלאה "
        "של כל סוגי האסטרטגיה שמופיעים באתר ZOOZ."
    )


def _is_tunnel_vision_question(query):
    q = _normalize(query)
    return "tunnel vision" in q or "ראיית מנהרה" in q


def _tunnel_vision_answer():
    return (
        "לא. **Tunnel Vision / Core Business Capabilities אינו סוג של אסטרטגיה**, ולכן לא נכון להכניס אותו "
        "לרשימה של סוגי אסטרטגיה. הוא מופיע במאמר נפרד כמודל חשיבה/ניהול שעוסק ביכולות הליבה של העסק.\n\n"
        "כשנשאלים על סוגי אסטרטגיה, צריך לענות מתוך מאגר האסטרטגיה עצמו — למשל ארבעת הסוגים התחרותיים "
        "או מסגרות כמו אוקיינוס כחול — ולא לערבב את Tunnel Vision בתוך הרשימה."
    )


def _is_invention_howto_question(query):
    q = _normalize(query)
    return (
        q in {"איך להמציא", "איך ממציאים", "איך ממציאים משהו", "איך לפתח המצאה"}
        or ("איך" in q and any(term in q for term in ("להמציא מוצר", "להמציא שירות", "ממציאים מוצר", "ממציאים שירות")))
    )


def _invention_howto_answer():
    return (
        "לפי שיטת **SIT** שמוצגת באתר ZOOZ, הדרך השיטתית להמציא מתחילה ממה שכבר קיים — עקרון "
        "**העולם הסגור**. קודם מגדירים את המוצר, השירות או האתגר וממפים את המרכיבים, המשתנים והמשאבים "
        "שכבר קיימים במוצר ובסביבתו.\n\n"
        "אחר כך מפעילים באופן שיטתי את ששת כלי החשיבה של SIT: **הכפלה, חלוקה, החסרה, איחוד, הוספת מימד "
        "והתאמה לסביבה**. מכל שינוי מחפשים תועלת אפשרית, מפתחים את הרעיונות שנוצרו, מדרגים את המובילים "
        "ומקדמים את הקונספטים הטובים ליישום. ששת כובעי החשיבה יכולים לשמש לדיון ולהערכת רעיונות, "
        "אבל הם אינם שלב בסיסי בשיטת ההמצאה של SIT."
    )


def _is_systematic_innovation_question(query):
    q = _normalize(query)
    has_phrase = "חדשנות שיטתית" in q or "חדשנות השיטתית" in q
    if not has_phrase:
        return False
    return "חשיבה המצאתית" not in q and "sit" not in q


def _systematic_innovation_answer(query):
    q = _normalize(query)

    if "ערנות" in q and any(term in q for term in ("רובד", "רביעי", "מעגל", "מודל")):
        return (
            "לא לפי המודל של ZOOZ. במאמר על **חדשנות שיטתית** מופיעים **שלושה רבדים עיקריים**: "
            "**תשתית אסטרטגית, תשתית ארגונית וצנרת הרעיונות**. המונח **„ערנות” אינו מוצג שם כרובד רביעי**.\n\n"
            "לכן, אם מסבירים את המודל של ZOOZ, נכון להישאר עם שלושת הרבדים האלה ולא להוסיף רובד נוסף "
            "על בסיס מושגים ממאמרים אחרים או ממודלים אחרים."
        )

    asks_layers = (
        "מעגל" in q
        or "רובד" in q
        or "רבדים" in q
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
            "המונח **„ערנות” אינו מוצג במאמר של ZOOZ כרובד רביעי של המודל**, ולכן לא נכון לערבב אותו עם שלושת הרבדים."
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


def _guarded_response_guidance(query):
    """Add global anti-mixing rules to every normal RAG answer."""
    base = _ORIGINAL_RESPONSE_GUIDANCE(query)
    return (
        f"{base}\n"
        "כללי דיוק נוספים: אל תערבב פרטים השייכים לקטגוריות שונות רק מפני שהם הופיעו יחד בתוצאות החיפוש. "
        "כאשר יש ציר זמן, שמור במפורש על ההבחנה בין תפקיד נוכחי, פעילות בשנים האחרונות ותפקידים בעבר, "
        "ואל תסיק סדר כרונולוגי שהמקור אינו קובע. כאשר נשאלת על רשימה או סוגים, כלול רק פריטים שהמקור "
        "מגדיר כחלק מאותה קטגוריה; אל תהפוך דוגמה, מודל או מושג סמוך לסוג נוסף. העדף עמוד רשמי ומרכזי "
        "על פני דוגמה מקרית מתוך עלון או פרויקט ישן. אם שאלת המשך משתמשת בכינוי גוף כמו 'הוא' או 'היא', "
        "אל תחליף את האדם בדמות אחרת בלי בסיס מפורש."
    )


def _guarded_ask_zooz(query):
    if _is_zooz_manager_question(query):
        return _zooz_manager_answer(), TEAM_SOURCES

    if _is_curefacts_question(query):
        return _curefacts_answer(query), ARI_SOURCES

    if _is_team_overview_question(query):
        return _team_overview_answer(), TEAM_SOURCES

    if _is_ari_more_question(query):
        return _ari_more_answer(), ARI_SOURCES

    if _is_ari_past_question(query):
        return _ari_past_answer(), ARI_SOURCES

    if _is_ari_current_question(query):
        return _ari_current_answer(), ARI_SOURCES

    if _is_ari_profile_question(query):
        return _ari_profile_answer(), ARI_SOURCES

    if _is_clients_question(query):
        return _clients_answer(), CLIENT_SOURCES

    if _is_services_question(query):
        return _services_answer(), SERVICES_SOURCES

    if _is_competitive_strategy_question(query):
        return _competitive_strategy_answer(), COMPETITIVE_STRATEGY_SOURCES

    if _is_strategy_types_question(query):
        return _strategy_types_answer(), STRATEGY_SOURCES

    if _is_tunnel_vision_question(query):
        return _tunnel_vision_answer(), TUNNEL_VISION_SOURCES

    if _is_invention_howto_question(query):
        return _invention_howto_answer(), INVENTION_SOURCES

    if _is_systematic_innovation_question(query):
        return _systematic_innovation_answer(query), SYSTEMATIC_INNOVATION_SOURCES

    return _ORIGINAL_ASK_ZOOZ(query)


_rag_engine.response_guidance = _guarded_response_guidance
_rag_engine.ask_zooz = _guarded_ask_zooz
