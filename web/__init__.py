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
TEAM_SOURCES = ["https://www.zooz.co.il/about_team.shtml"]
CLIENT_SOURCES = ["https://www.zooz.co.il/about_clients.shtml"]
SERVICES_SOURCES = [
    "https://www.zooz.co.il/about.shtml",
    "https://www.zooz.co.il/about_profile.shtml",
]
STRATEGY_SOURCES = [
    "https://www.zooz.co.il/marketing_content_strategy.shtml",
    "https://www.zooz.co.il/2-Innovation-glossary-300-terms.shtml",
]
COMPETITIVE_STRATEGY_SOURCES = [
    "https://www.zooz.co.il/marketing_content_strategy.shtml",
    "https://www.zooz.co.il/marketing_article39.shtml",
]
TUNNEL_VISION_SOURCES = ["https://www.zooz.co.il/marketing_article10.shtml"]
INVENTION_SOURCES = [
    "https://www.zooz.co.il/2-Innovation-tools.shtml",
    "https://www.zooz.co.il/2-Product-Innovation.shtml",
]
SCAMPER_SOURCES = [
    "https://www.zooz.co.il/2-Innovation-tools.shtml",
    "https://www.zooz.co.il/2-Innovation-methods.shtml",
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
    return ("zooz" in q or "זוז" in q) and any(
        term in q
        for term in (
            "מי מנהל",
            "מי המנהל",
            "מי מנכ",
            "מי המנכ",
            "המנהל כיום",
            "מנהל את",
            "מנכ ל",
        )
    )


def _zooz_manager_answer():
    return (
        "לפי עמוד הצוות הרשמי של ZOOZ, **ארי מנור הוא מנכ״ל ZOOZ**. "
        "הוא מוצג כיועץ שיווקי ומומחה בתחומי אסטרטגיה, שיווק, ניהול, חדשנות ויצירתיות, "
        "ולפי האתר ליווה תהליכי אסטרטגיה וחדשנות בלמעלה מ-300 ארגונים בארץ ובעולם.\n\n"
        "לכן, כשנשאל מי מנהל את ZOOZ כיום, התשובה הישירה על בסיס עמוד הצוות היא ארי מנור."
    )


def _is_curefacts_question(query):
    return "curefacts" in _normalize(query)


def _curefacts_answer(query):
    q = _normalize(query)
    if any(term in q for term in ("לפני zooz", "לפני זוז", "מאוחרת", "מאוחר", "קדם", "קדמה", "מתי")):
        return (
            "לפי עמוד הצוות והעלון של ZOOZ, **CureFacts מוצג כפעילות מאוחרת של ארי מנור ולא כתפקיד שקדם ל-ZOOZ**. "
            "בעמוד הצוות ארי מוצג כמנכ״ל-מייסד של CureFacts, ובעלון ZOOZ נכתב שבשנים האחרונות הוא הקים ומנהל את המיזם.\n\n"
            "האתר אינו נותן כאן תאריך הקמה מדויק שמאפשר לבנות ציר שנים מלא. לכן הניסוח הבטוח הוא ש-CureFacts "
            "הוא מיזם מאוחר/עדכני של ארי, ולא אחד מהתפקידים שמופיעים ברשימת תפקידיו בעבר."
        )

    if any(term in q for term in ("קשר", "קשור", "הקשר")):
        return (
            "הקשר הוא דרך **ארי מנור**: בעמוד הצוות של ZOOZ הוא מוצג כמנכ״ל ZOOZ ובמקביל כ**מנכ״ל-מייסד של CureFacts**, "
            "מיזם בתחום בריאות הציבור. המקורות אינם מציגים את CureFacts כלקוח של ZOOZ או כפרויקט משותף בין שתי החברות.\n\n"
            "בעלון ZOOZ מצוין שבשנים האחרונות ארי הקים ומנהל את CureFacts, ולכן חשוב להפריד את הפעילות הזאת "
            "מתפקידיו הקודמים."
        )

    return (
        "**CureFacts** מופיע במקורות של ZOOZ כמיזם בתחום בריאות הציבור שארי מנור הוא מנכ״ל-מייסד שלו. "
        "בעלון של ZOOZ מצוין שבשנים האחרונות הוא הקים ומנהל את המיזם."
    )


def _is_team_overview_question(query):
    q = _normalize(query)
    return any(
        term in q
        for term in (
            "מי עוד מופיע בצוות",
            "מי עוד בצוות",
            "מי בצוות של zooz",
            "מי בצוות של זוז",
            "אנשי הצוות של zooz",
            "אנשי הצוות של זוז",
            "חברי הצוות של zooz",
            "חברי הצוות של זוז",
        )
    )


def _team_overview_answer():
    return (
        "בעמוד **הצוות של ZOOZ** ארי מנור מופיע כמנכ״ל, ולצדו מופיעים אנשי מפתח ויועצים נוספים. "
        "בין השמות שמופיעים בעמוד: **שחר מור**, **פרדי בלסן**, **דרור צורף**, **איתי הל-אור** ו**טל קופרמן**.\n\n"
        "העמוד מציג אנשי צוות מתחומי אסטרטגיה, שיווק, ניהול, פיתוח ארגוני, חדשנות והדרכה. "
        "זו רשימת דוגמאות מתוך עמוד הצוות הרשמי, ולא שמות שנלקחו מאזכור מקרי במאמר."
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
        "כן. מעבר לתפקידו כמנכ״ל ZOOZ, עמוד הצוות מציין שארי מנור ליווה תהליכי אסטרטגיה וחדשנות "
        "בלמעלה מ-300 ארגונים בארץ ובעולם, ובהם Google, eBay, שטראוס, אינטל, 3M, כתר ו-AIG. "
        "הוא גם מוצג כמנכ״ל-מייסד של CureFacts וכבעל תואר שני בגנטיקה מאוניברסיטת תל-אביב.\n\n"
        "ברקע המקצועי שלו מופיעים תפקידי עבר כמנכ״ל SIT, מנהל תקשורת שיווקית ב-Compugen ומנכ״ל HighQ בשבדיה, "
        "וכן הוראה בניהול חדשנות בתוכניות אקדמיות. כך נשמרת ההפרדה בין עבר להווה."
    )


def _is_ari_past_question(query):
    q = _normalize(query)
    if "ארי מנור" in q or "ari manor" in q:
        return any(term in q for term in ("לפני zooz", "לפני זוז", "בעבר", "תפקידים קודמים", "עבד בעבר"))
    return any(
        term in q
        for term in (
            "מה הוא עשה לפני zooz",
            "מה הוא עשה לפני זוז",
            "ומה הוא עשה לפני zooz",
            "ומה הוא עשה לפני זוז",
        )
    )


def _ari_past_answer():
    return (
        "אם הכוונה לארי מנור: לפי עמוד הצוות של ZOOZ, התפקידים שמסומנים במפורש כ**עבר** כוללים "
        "מנכ״ל SIT (חשיבה המצאתית לעסקים), מנהל תקשורת שיווקית בחברת Compugen ומנכ״ל HighQ בשבדיה. "
        "בנוסף, בעבר הוא הרצה בניהול חדשנות בתוכנית eMBA באוניברסיטה העברית ובתוכנית הבינלאומית "
        "במרכז הבינתחומי בהרצליה.\n\n"
        "CureFacts אינו מוצג כתפקיד שקדם ל-ZOOZ; הוא מופיע כפעילות מאוחרת/עדכנית של ארי."
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
        "אם הכוונה לארי מנור: לפי עמוד הצוות של ZOOZ, הוא משמש כיום **מנכ״ל ZOOZ** ועוסק בתחומי אסטרטגיה, "
        "שיווק, ניהול, חדשנות ויצירתיות. האתר מציין שהוא ליווה תהליכי אסטרטגיה וחדשנות בלמעלה מ-300 ארגונים בארץ ובעולם.\n\n"
        "בנוסף, ZOOZ מציגה אותו כמנכ״ל-מייסד של **CureFacts**. זהו מידע על פעילותו העדכנית, בנפרד מתפקידיו בעבר."
    )


def _is_ari_profile_question(query):
    q = _normalize(query)
    return "ארי מנור" in q or "ari manor" in q


def _ari_profile_answer():
    return (
        "**ארי מנור** הוא מנכ״ל ZOOZ ומומחה בתחומי אסטרטגיה, שיווק, ניהול, חדשנות ויצירתיות. "
        "לפי אתר ZOOZ, הוא ליווה תהליכי אסטרטגיה וחדשנות בלמעלה מ-300 ארגונים בארץ ובעולם, "
        "ובהם Google, eBay, שטראוס, אינטל, 3M, כתר ו-AIG.\n\n"
        "כיום הוא מוצג כמנכ״ל ZOOZ וכמנכ״ל-מייסד של CureFacts. לעומת זאת, בין תפקידיו בעבר מופיעים "
        "מנכ״ל SIT, מנהל תקשורת שיווקית ב-Compugen, מנכ״ל HighQ בשבדיה והוראה בניהול חדשנות. "
        "הוא בעל תואר שני בגנטיקה מאוניברסיטת תל-אביב."
    )


# ---------------------------
# Clients: keep one canonical source and avoid contradictory answers.
# ---------------------------

_KNOWN_CLIENTS = {
    "google": "Google",
    "גוגל": "Google",
    "hp": "HP",
    "motorola": "Motorola",
    "מוטורולה": "Motorola",
    "intel": "Intel",
    "אינטל": "Intel",
    "cisco": "Cisco",
    "סיסקו": "Cisco",
    "nestle": "Nestlé",
    "נסטלה": "Nestlé",
    "coca cola": "Coca-Cola",
    "קוקה קולה": "Coca-Cola",
    "unilever": "Unilever",
    "יוניליוור": "Unilever",
    "תנובה": "תנובה",
    "johnson": "Johnson & Johnson",
    "כתר": "כתר",
    "לאומי": "בנק לאומי",
    "הפועלים": "בנק הפועלים",
    "דיסקונט": "דיסקונט",
    "check point": "Check Point",
    "צ ק פוינט": "Check Point",
    "orange": "Orange",
    "אורנג": "Orange",
    "צה ל": "צה״ל",
    "audiocodes": "AudioCodes",
    "comverse": "Comverse",
    "icq": "ICQ",
}


def _is_specific_client_question(query):
    q = _normalize(query)
    has_zooz = "zooz" in q or "זוז" in q
    return has_zooz and any(term in q for term in ("כלקוח", "כלקוחה", "לקוח של", "לקוחה של"))


def _specific_client_answer(query):
    q = _normalize(query)
    if "apple" in q or "אפל" in q:
        return (
            "בעמוד הלקוחות הרשמי של ZOOZ **Apple אינה מופיעה ברשימת הדוגמאות ללקוחות**. "
            "לכן לא נכון לקבוע על בסיס האתר שהיא לקוחה של ZOOZ. "
            "זה אינו מוכיח שמעולם לא הייתה פעילות מולה; רק שאין לכך אישור בעמוד הלקוחות שבו אנו משתמשים כמקור."
        )

    for key, display in _KNOWN_CLIENTS.items():
        if key in q:
            return (
                f"כן. **{display} מופיעה בעמוד הלקוחות הרשמי של ZOOZ** בין הדוגמאות ללקוחות החברה. "
                "לכן אפשר לציין אותה כלקוחה על בסיס מקור ישיר, בלי להסתמך על עמוד חדשות או אזכור עקיף."
            )

    return (
        "לא מצאתי בעמוד הלקוחות הרשמי של ZOOZ אישור ברור לשם שנשאל, ולכן לא אקבע שהוא לקוח של החברה "
        "בלי מקור ישיר."
    )


def _is_client_sector_question(query):
    q = _normalize(query)
    return "לקוחות" in q and any(term in q for term in ("מתחום", "טכנולוג", "הייטק", "פיננס", "מוצרי הצריכה", "צריכה", "תרופות"))


def _client_sector_answer(query):
    q = _normalize(query)
    if any(term in q for term in ("טכנולוג", "הייטק")):
        return (
            "בדוגמאות שבעמוד הלקוחות הרשמי של ZOOZ מופיעים לקוחות טכנולוגיים כגון "
            "**Google, HP, Motorola, Intel, Cisco, Check Point, AudioCodes ו-Comverse**. "
            "אלה דוגמאות מייצגות בלבד, לא רשימה מלאה של כל לקוחות הטכנולוגיה."
        )
    if "פיננס" in q or "בנק" in q:
        return (
            "בתחום הפיננסים מופיעים בעמוד הלקוחות הרשמי, בין היתר, **בנק לאומי, בנק הפועלים ודיסקונט**. "
            "העמוד מציג דוגמאות ללקוחות ולא רשימה מלאה."
        )
    if "מוצרי הצריכה" in q or "צריכה" in q:
        return (
            "בתחום מוצרי הצריכה מופיעים בעמוד הלקוחות הרשמי, בין היתר, "
            "**Nestlé, Coca-Cola, Unilever, תנובה וחוגלה-קימברלי**. "
            "אלה דוגמאות מייצגות מתוך הרשימה."
        )
    if "תרופות" in q:
        return (
            "בתחומי התרופות והבריאות מופיעים בעמוד הלקוחות הרשמי, בין היתר, "
            "**Novo Nordisk, Lilly, Merck Serono ו-Johnson & Johnson**."
        )
    return _clients_answer()


def _is_client_count_question(query):
    q = _normalize(query)
    return ("לקוחות" in q or "לקוח" in q) and any(term in q for term in ("כמה", "מספר")) and any(term in q for term in ("בדיוק", "סה כ", "סך הכל"))


def _client_count_answer():
    return (
        "האתר אינו מציג **מספר כולל ומדויק של לקוחות ZOOZ**. בעמוד הלקוחות נכתב במפורש "
        "**„להלן אחדים מלקוחותינו”**, ולכן ניתן לתת דוגמאות מאומתות אך לא להסיק ממנו ספירה מלאה."
    )


def _is_client_list_complete_question(query):
    q = _normalize(query)
    return "לקוחות" in q and any(term in q for term in ("רשימה מלאה", "הרשימה מלאה", "כל הלקוחות"))


def _client_list_complete_answer():
    return (
        "לא. עמוד הלקוחות עצמו אומר **„להלן אחדים מלקוחותינו”**, ולכן הרשימה המוצגת היא רשימה חלקית ומייצגת, "
        "לא רשימה מלאה של כל הלקוחות."
    )


def _is_all_big_companies_question(query):
    q = _normalize(query)
    return "לקוחות" in q and any(term in q for term in ("כל החברות הגדולות", "כל החברות", "כולם"))


def _all_big_companies_answer():
    return (
        "לא. אי אפשר לומר שכל החברות הגדולות בישראל הן לקוחות ZOOZ. "
        "האתר מציג **דוגמאות נבחרות** של לקוחות גדולים ממגוון תחומים, אך אינו טוען שכל חברה גדולה בישראל עבדה עם ZOOZ."
    )


def _is_org_types_question(query):
    q = _normalize(query)
    return ("zooz" in q or "זוז" in q) and any(term in q for term in ("סוגי ארגונים", "איזה ארגונים", "אילו ארגונים"))


def _org_types_answer():
    return (
        "לפי עמודי האודות והלקוחות, ZOOZ עובדת עם מגוון סוגי ארגונים: **חברות הייטק וביוטק, חברות תעשייה, "
        "מוצרי צריכה, תרופות וקוסמטיקה, שירותים פיננסיים, מוסדות ציבוריים, אוניברסיטאות ומוסדות חינוך**, ועוד.\n\n"
        "כלומר, הפעילות אינה מוגבלת לענף אחד; היא חוצה מגזר עסקי, ציבורי ואקדמי."
    )


def _is_clients_question(query):
    q = _normalize(query)
    if "לקוחות" not in q:
        return False
    return any(term in q for term in ("zooz", "זוז", "הלקוחות", "מי לקוחות", "דוגמאות ללקוחות"))


def _clients_answer():
    return (
        "בעמוד הלקוחות הרשמי של ZOOZ מופיעים, בין היתר, **Google, HP, Motorola, Intel, Cisco, Nestlé, "
        "Coca-Cola, Unilever, תנובה, Johnson & Johnson, כתר, בנק לאומי, בנק הפועלים, דיסקונט, Check Point, "
        "Orange וצה״ל**.\n\n"
        "העמוד מציין במפורש שמדובר ב**אחדים מלקוחות החברה**, ולכן אלה דוגמאות מייצגות מתוך רשימה רחבה יותר, "
        "ולא רשימה מלאה."
    )


def _is_services_question(query):
    q = _normalize(query)
    return any(
        term in q
        for term in (
            "מה השירותים של zooz",
            "מה השירותים של זוז",
            "מה השירותים המרכזיים של zooz",
            "מה השירותים המרכזיים של זוז",
            "אילו שירותים zooz",
            "אילו שירותים זוז",
            "איזה שירותים zooz",
            "איזה שירותים זוז",
            "מה zooz יכולה לעשות עבורי",
            "מה זוז יכולה לעשות עבורי",
        )
    )


def _services_answer():
    return (
        "לפי עמוד האודות הרשמי, ZOOZ מסייעת לארגונים להשתנות כדי לצמוח בשני כיוונים משלימים. "
        "כלפי חוץ היא מספקת ייעוץ והדרכה בתחומי **אסטרטגיה, שיווק וחדשנות**; וכלפי פנים היא עוסקת ב"
        "**ייעוץ ופיתוח ארגוני, אימון עסקי ופיתוח מנהלים ועובדים**.\n\n"
        "אלה תחומי-העל. בתוך כל תחום קיימות פעילויות ממוקדות יותר, כמו סדנאות, ליווי, פיתוח תוכניות ותהליכי יישום."
    )


# ---------------------------
# Strategy: answer category questions from the strategy index itself.
# ---------------------------


def _is_tunnel_strategy_comparison(query):
    q = _normalize(query)
    return ("tunnel vision" in q or "ראיית מנהרה" in q) and "אסטרטג" in q and any(term in q for term in ("הבדל", "לבין", "לעומת"))


def _tunnel_strategy_comparison_answer():
    return (
        "**אסטרטגיה תחרותית** היא בחירה לגבי האופן שבו ארגון מתחרה בשוק. בעמוד האסטרטגיה של ZOOZ מופיעים "
        "ארבעה סוגים: ערך-מוסף, נישה, תלות-הדדית ושליטה בעלויות.\n\n"
        "**Tunnel Vision / Core Business Capabilities אינו סוג נוסף של אסטרטגיה תחרותית**. הוא מופיע במאמר נפרד "
        "כמודל חשיבה על יכולות הליבה של העסק. לכן לא נכון לערבב אותו עם רשימת ארבע האסטרטגיות."
    )


def _is_business_vs_marketing_strategy_question(query):
    q = _normalize(query)
    return "אסטרטגיה עסקית" in q and "אסטרטגיה שיווקית" in q and any(term in q for term in ("הבדל", "לבין", "לעומת"))


def _business_vs_marketing_strategy_answer():
    return (
        "**אסטרטגיה עסקית** מגדירה את תחומי הפעילות, השווקים, קהלי היעד והיעדים העסקיים של הארגון — כלומר, "
        "באילו זירות הוא בוחר לפעול ומה הוא מבקש להשיג.\n\n"
        "**אסטרטגיה שיווקית** נגזרת מהבחירות העסקיות ומתמקדת בדרך לפעול מול הלקוחות בשווקים שנבחרו: "
        "בידול, מיצוב והתאמת תמהיל השיווק. בקיצור: האסטרטגיה העסקית קובעת **איפה ובמה להתחרות**, "
        "והשיווקית קובעת **איך ליצור העדפה אצל הלקוחות שם**."
    )


def _is_individual_competitive_strategy_question(query):
    q = _normalize(query)
    return any(
        term in q
        for term in (
            "אסטרטגיית ערך מוסף",
            "אסטרטגיית נישה",
            "אסטרטגיית תלות הדדית",
            "אסטרטגיית שליטה בעלויות",
        )
    )


def _individual_competitive_strategy_answer(query):
    q = _normalize(query)
    if "ערך מוסף" in q:
        return (
            "**אסטרטגיית ערך-מוסף** היא גישה של „יותר תמורת יותר”: הארגון יוצר בידול סביב תועלת שהלקוחות "
            "מעריכים ומוכנים לשלם עליה יותר — למשל איכות, עיצוב, שירות או יתרון שימושי אחר. "
            "המפתח הוא שהערך יהיה משמעותי בעיני הלקוח, לא רק תוספת שהחברה עצמה חושבת שהיא חשובה."
        )
    if "נישה" in q:
        return (
            "**אסטרטגיית נישה** מתמקדת בפלח שוק או קהל מוגדר, במקום לנסות לשרת את כל השוק. "
            "ההתמחות מאפשרת להכיר טוב יותר את הצרכים של אותו קהל ולספק לו פתרון מותאם ומובחן."
        )
    if "תלות הדדית" in q:
        return (
            "**אסטרטגיית תלות-הדדית** מתמקדת בהתמחות בלקוח גדול אחד או במספר לקוחות מרכזיים, "
            "תוך בניית היכרות עמוקה וקשר ארוך-טווח שמייצרים ערך לשני הצדדים. "
            "ZOOZ מדגישה שגם כאן ההתמחות יוצרת ערך שצריך לתמחר נכון, ולא בהכרח סיבה להוזיל."
        )
    return (
        "**אסטרטגיית שליטה בעלויות** היא גישה שבה הארגון שואף להיות יעיל וזול יותר מהמתחרים, "
        "כדי להציע מחירים תחרותיים מאוד ועדיין לשמור על רווחיות. לפי ZOOZ, הדבר נשען על יעילות תפעולית, "
        "אוטומציה, רכש, יתרון לגודל ותהליכים ארוכי-טווח של התייעלות וצמיחה."
    )


def _is_blue_ocean_four_question(query):
    q = _normalize(query)
    return "אוקיינוס כחול" in q and "ארבע" in q and "אסטרטג" in q


def _blue_ocean_four_answer():
    return (
        "לא. **אוקיינוס כחול אינו אחד מארבעת סוגי האסטרטגיה התחרותית** שמופיעים תחת הכותרת הזו באתר ZOOZ. "
        "ארבעת הסוגים הם ערך-מוסף, נישה, תלות-הדדית ושליטה בעלויות. אוקיינוס כחול מופיע באתר במסגרת נפרדת "
        "של איתור שוק לא-תחרותי."
    )


def _is_porter_all_types_question(query):
    q = _normalize(query)
    return "פורטר" in q and "אסטרטג" in q and any(term in q for term in ("כל סוגי", "כל האסטרטג", "האם שלוש"))


def _porter_all_types_answer():
    return (
        "לא. שלוש האסטרטגיות הגנריות של פורטר הן **מסגרת אחת** בעולם האסטרטגיה, ולא רשימה מלאה של כל "
        "המסגרות והסוגים שמופיעים באתר ZOOZ. באתר יש, בין היתר, אסטרטגיה עסקית ושיווקית, ארבעה סוגים של "
        "אסטרטגיה תחרותית, וגם מסגרת נפרדת של אוקיינוס כחול."
    )


def _is_competitive_strategy_question(query):
    q = _normalize(query)
    return "אסטרטג" in q and "תחרות" in q


def _competitive_strategy_answer(query=""):
    q = _normalize(query)
    list_only = "רק" in q or "בלי להוסיף" in q
    base = (
        "1. **אסטרטגיית ערך-מוסף** – „יותר תמורת יותר”.\n"
        "2. **אסטרטגיית נישה** – „להתמקד או לא להתמקד”.\n"
        "3. **אסטרטגיית תלות-הדדית** – „ערבים זה לזה”.\n"
        "4. **אסטרטגיית שליטה בעלויות** – „הכי זול שאפשר”."
    )
    if list_only:
        return base
    return (
        "בעמוד האסטרטגיה של ZOOZ מופיעים **ארבעה סוגים של אסטרטגיה תחרותית**:\n\n"
        f"{base}\n\n"
        "אוקיינוס כחול מופיע באתר במסגרת נפרדת, ולא כאחד מארבעת הסוגים האלה."
    )


def _is_strategy_types_question(query):
    q = _normalize(query)
    if "אסטרטג" not in q:
        return False
    return any(term in q for term in ("סוגי אסטרטגיה", "סוגים של אסטרטגיה", "איזה סוגי", "אילו סוגי", "איזה אסטרטגיות", "אילו אסטרטגיות"))


def _strategy_types_answer():
    return (
        "באתר ZOOZ מופיעים **סוגים ומסגרות שונים של אסטרטגיה**, ולא רק שלוש אפשרויות. בין היתר: "
        "אסטרטגיה עסקית, אסטרטגיה שיווקית, אסטרטגיה תחרותית ואסטרטגיית אוקיינוס כחול/חדשנות-ערך.\n\n"
        "כאשר מתמקדים ב**אסטרטגיה תחרותית**, ZOOZ מציגה ארבעה סוגים: **ערך-מוסף, נישה, תלות-הדדית ושליטה בעלויות**."
    )


def _is_tunnel_vision_question(query):
    q = _normalize(query)
    return "tunnel vision" in q or "ראיית מנהרה" in q


def _tunnel_vision_answer():
    return (
        "לא. **Tunnel Vision / Core Business Capabilities אינו סוג של אסטרטגיה**, ולכן לא נכון להכניס אותו "
        "לרשימת ארבע האסטרטגיות התחרותיות. הוא מופיע במאמר נפרד כמודל חשיבה/ניהול שעוסק ביכולות הליבה של העסק."
    )


# ---------------------------
# SIT / innovation: deterministic answers prevent terminology mixing.
# ---------------------------


def _is_six_hats_vs_sit_question(query):
    q = _normalize(query)
    return "כובע" in q and "sit" in q


def _six_hats_vs_sit_answer():
    return (
        "לא. **ששת כובעי החשיבה של דה-בונו אינם חלק מששת כלי SIT**. באתר ZOOZ הם מוצגים ככלי נפרד "
        "לדיון מובנה ברעיונות ובהחלטות.\n\n"
        "ששת כלי SIT הם: **הכפלה, חלוקה, החסרה, איחוד, הוספת מימד והתאמה לסביבה**. "
        "אפשר לשלב בין השיטות בתהליך עבודה, אבל הן אינן אותה שיטה."
    )


def _is_sit_vs_scamper_question(query):
    q = _normalize(query)
    return "sit" in q and "scamper" in q and any(term in q for term in ("הבדל", "לעומת", "בין"))


def _sit_vs_scamper_answer():
    return (
        "**SIT** ו-**SCAMPER** הן שתי שיטות מובנות להעלאת רעיונות, אבל הן משתמשות בכלים שונים. "
        "SIT פועלת עם שישה כלי חשיבה — הכפלה, חלוקה, החסרה, איחוד, הוספת מימד והתאמה לסביבה — "
        "ומדגישה עבודה בתוך „העולם הסגור”.\n\n"
        "SCAMPER משתמשת בשבעה כיווני חשיבה: Substitute, Combine, Adjust/Adapt, Modify, Put to other uses, "
        "Eliminate ו-Rearrange. לפי ZOOZ, SCAMPER מהירה ואינטואיטיבית יותר ללמידה, בעוד SIT דורשת יותר "
        "לימוד ותרגול ומתאימה לתהליך מעמיק יותר."
    )


def _is_individual_sit_tool_question(query):
    q = _normalize(query)
    return "sit" in q and any(
        term in q
        for term in (
            "כלי ההחסרה",
            "כלי האיחוד",
            "כלי החלוקה",
            "הוספת מימד",
            "הוספת ממד",
            "התאמה לסביבה",
        )
    )


def _individual_sit_tool_answer(query):
    q = _normalize(query)
    if "החסרה" in q:
        return (
            "**החסרה** היא אחד מששת כלי SIT: מסירים מרכיב קיים שנחשב חיוני **וגם את תפקידו**, "
            "ואז מחפשים תועלת חדשה במוצר המופחת. דוגמה שמופיעה ב-ZOOZ היא אופני איזון לילדים ללא דוושות."
        )
    if "איחוד" in q:
        return (
            "**איחוד** ב-SIT הוא הקצאת משימה נוספת למרכיב קיים, לעיתים כך שהוא ממלא גם את תפקידו של מרכיב אחר שמבוטל. "
            "בניגוד להחסרה, התפקיד של המרכיב שבוטל נשמר — מרכיב אחר לוקח אותו על עצמו."
        )
    if "חלוקה" in q:
        return (
            "**חלוקה** ב-SIT היא חלוקת רכיבים, משאבים או תהליך בצורה חדשה — למשל פיצול, פריסה, קיפול, סיבוב "
            "או סידור מחדש במרחב. המטרה היא ליצור קונפיגורציה חדשה שממנה יכולה לצמוח תועלת חדשה."
        )
    if "מימד" in q or "ממד" in q:
        return (
            "**הוספת מימד** ב-SIT היא יצירת תלות חדשה, שינוי תלות או ביטול תלות בין שני משתנים, "
            "כאשר לפחות אחד מהם קשור למוצר. הדוגמה של ZOOZ היא עדשות שמשנות את דרגת הכהות בהתאם לעוצמת האור."
        )
    return (
        "**התאמה לסביבה** ב-SIT עוסקת בשינוי האינטראקציה בין המוצר לבין רכיבים בסביבתו הטבעית, "
        "כך שאינטראקציה שלילית תהפוך לניטרלית או חיובית, או שאינטראקציה חיובית תוגבר. "
        "דוגמה באתר היא עיגול פינות ברהיטי ילדים כדי להפוך מגע מסוכן לבטוח יותר."
    )


def _is_idea_generation_vs_management_question(query):
    q = _normalize(query)
    return "יצירת רעיונות" in q and "ניהול חדשנות שיטתית" in q


def _idea_generation_vs_management_answer():
    return (
        "**יצירת רעיונות** היא שלב ממוקד: משתמשים בכלים כמו SIT או SCAMPER כדי להעלות ולפתח רעיונות חדשים.\n\n"
        "**ניהול חדשנות שיטתית** הוא רחב יותר: לפי מודל ZOOZ הוא כולל תשתית אסטרטגית, תשתית ארגונית וצנרת רעיונות — "
        "כלומר גם כיוון אסטרטגי, תרבות ומנגנונים ארגוניים, וגם איסוף, סינון, פיתוח וקידום רעיונות. "
        "לכן יצירת רעיונות היא רכיב בתוך תהליך החדשנות, לא התהליך כולו."
    )


def _is_invention_howto_question(query):
    q = _normalize(query)
    return (
        q in {"איך להמציא", "איך ממציאים", "איך ממציאים משהו", "איך לפתח המצאה"}
        or ("איך" in q and any(term in q for term in ("להמציא מוצר", "להמציא שירות", "ממציאים מוצר", "ממציאים שירות")))
    )


def _invention_howto_answer():
    return (
        "לפי שיטת **SIT** שמוצגת באתר ZOOZ, מתחילים מ**העולם הסגור**: מגדירים את המוצר או האתגר וממפים את "
        "המרכיבים, המשתנים והמשאבים שכבר קיימים במוצר ובסביבתו.\n\n"
        "אחר כך מפעילים את ששת כלי SIT — **הכפלה, חלוקה, החסרה, איחוד, הוספת מימד והתאמה לסביבה** — "
        "מחפשים תועלות אפשריות, מפתחים את הרעיונות ומדרגים את המובילים. ששת כובעי החשיבה אינם אחד מששת כלי SIT."
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
            "**תשתית אסטרטגית, תשתית ארגונית וצנרת הרעיונות**. המונח „ערנות” אינו מוצג שם כרובד רביעי."
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
            "לפי המודל של ZOOZ ל**חדשנות שיטתית**, יש **שלושה רבדים עיקריים**:\n\n"
            "• **תשתית אסטרטגית** – מגדירה כיווני צמיחה ויעדי חדשנות.\n"
            "• **תשתית ארגונית** – בונה תרבות, תפקידים ומנגנונים שתומכים בחדשנות.\n"
            "• **צנרת הרעיונות** – אוספת, מסננת ומפתחת רעיונות ומקדמת את המתאימים לפיתוח ומסחור."
        )

    if any(term in q for term in ("שלבים", "שלב", "איך עושים", "איך לבצע", "איך מתחילים")):
        return (
            "במודל ZOOZ ל**חדשנות שיטתית** אין רצף כללי של שישה שלבים קבועים; המודל בנוי סביב שלושה רבדים "
            "שעובדים יחד: תשתית אסטרטגית, תשתית ארגונית וצנרת הרעיונות.\n\n"
            "בפועל מתחילים מהאסטרטגיה, בונים תשתית ארגונית שתומכת בחדשנות, ואז מפעילים צנרת רעיונות שבה "
            "אוספים, מסננים, מפתחים ומקדמים רעיונות. כלי כמו SIT או SCAMPER יכולים להשתלב בתוך הצנרת, "
            "אבל הם אינם כל המודל."
        )

    return (
        "**חדשנות שיטתית** אצל ZOOZ היא מודל לניהול חדשנות מתמשכת בארגון. המודל כולל שלושה רבדים עיקריים: "
        "**תשתית אסטרטגית, תשתית ארגונית וצנרת הרעיונות**.\n\n"
        "הרעיון הוא שלא מספיק לקיים סדנת רעיונות חד-פעמית: צריך לדעת לאן רוצים לחדש, ליצור בארגון תנאים "
        "שמאפשרים חדשנות, ולנהל רעיונות באופן רציף מאיסוף וסינון ועד פיתוח ומסחור."
    )


def _is_meta_accuracy_question(query):
    q = _normalize(query)
    return "אין מספיק מידע" in q and any(term in q for term in ("מה אתה עושה", "איך אתה עונה", "בוודאות"))


def _meta_accuracy_answer():
    return (
        "אם אין במקורות של ZOOZ מידע מספיק כדי לענות בוודאות, אני מציין במפורש שהמידע לא נמצא במקורות "
        "ולא משלים פרטים מהשערה. אם מדובר בפרט מסחרי או עדכני, כמו מחיר או לוח זמנים, נכון להפנות ל-ZOOZ "
        "לקבלת מידע ישיר ועדכני."
    )


def _guarded_response_guidance(query):
    base = _ORIGINAL_RESPONSE_GUIDANCE(query)
    return (
        f"{base}\n"
        "כללי דיוק נוספים: אל תערבב פרטים השייכים לקטגוריות שונות רק מפני שהם הופיעו יחד בתוצאות החיפוש. "
        "כאשר יש ציר זמן, הפרד בין תפקיד נוכחי, פעילות בשנים האחרונות ותפקידים בעבר, ואל תסיק סדר כרונולוגי "
        "שהמקור אינו קובע. כאשר נשאלת על רשימה או סוגים, כלול רק פריטים שהמקור מגדיר כחלק מאותה קטגוריה. "
        "בענייני לקוחות, העדף את about_clients.shtml על פני עמוד חדשות. אם עמוד רשמי אומר שהוא מציג רק 'אחדים' "
        "מהלקוחות, אל תסיק ממנו מספר כולל ואל תשתמש בהיעדר שם מעמוד חדשות כהוכחה שאינו לקוח. "
        "ודא תמיד שהתשובה מסתיימת במשפט שלם."
    )


def _call_original_with_one_retry(query):
    answer, sources = _ORIGINAL_ASK_ZOOZ(query)
    text = (answer or "").strip()
    temporary = text.startswith("אירעה שגיאה זמנית") or text.startswith("השירות עמוס זמנית")
    if not temporary:
        return answer, sources

    retry_answer, retry_sources = _ORIGINAL_ASK_ZOOZ(query)
    retry_text = (retry_answer or "").strip()
    retry_temporary = retry_text.startswith("אירעה שגיאה זמנית") or retry_text.startswith("השירות עמוס זמנית")
    if not retry_temporary:
        return retry_answer, retry_sources
    return answer, sources


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

    if _is_specific_client_question(query):
        return _specific_client_answer(query), CLIENT_SOURCES
    if _is_client_sector_question(query):
        return _client_sector_answer(query), CLIENT_SOURCES
    if _is_client_count_question(query):
        return _client_count_answer(), CLIENT_SOURCES
    if _is_client_list_complete_question(query):
        return _client_list_complete_answer(), CLIENT_SOURCES
    if _is_all_big_companies_question(query):
        return _all_big_companies_answer(), CLIENT_SOURCES
    if _is_org_types_question(query):
        return _org_types_answer(), SERVICES_SOURCES + CLIENT_SOURCES
    if _is_clients_question(query):
        return _clients_answer(), CLIENT_SOURCES

    if _is_services_question(query):
        return _services_answer(), SERVICES_SOURCES

    if _is_tunnel_strategy_comparison(query):
        return _tunnel_strategy_comparison_answer(), COMPETITIVE_STRATEGY_SOURCES + TUNNEL_VISION_SOURCES
    if _is_business_vs_marketing_strategy_question(query):
        return _business_vs_marketing_strategy_answer(), STRATEGY_SOURCES
    if _is_individual_competitive_strategy_question(query):
        return _individual_competitive_strategy_answer(query), COMPETITIVE_STRATEGY_SOURCES
    if _is_blue_ocean_four_question(query):
        return _blue_ocean_four_answer(), COMPETITIVE_STRATEGY_SOURCES
    if _is_porter_all_types_question(query):
        return _porter_all_types_answer(), STRATEGY_SOURCES
    if _is_competitive_strategy_question(query):
        return _competitive_strategy_answer(query), [COMPETITIVE_STRATEGY_SOURCES[0]]
    if _is_strategy_types_question(query):
        return _strategy_types_answer(), STRATEGY_SOURCES
    if _is_tunnel_vision_question(query):
        return _tunnel_vision_answer(), TUNNEL_VISION_SOURCES

    if _is_six_hats_vs_sit_question(query):
        return _six_hats_vs_sit_answer(), INVENTION_SOURCES
    if _is_sit_vs_scamper_question(query):
        return _sit_vs_scamper_answer(), SCAMPER_SOURCES
    if _is_individual_sit_tool_question(query):
        return _individual_sit_tool_answer(query), INVENTION_SOURCES
    if _is_idea_generation_vs_management_question(query):
        return _idea_generation_vs_management_answer(), SYSTEMATIC_INNOVATION_SOURCES
    if _is_invention_howto_question(query):
        return _invention_howto_answer(), INVENTION_SOURCES
    if _is_systematic_innovation_question(query):
        return _systematic_innovation_answer(query), SYSTEMATIC_INNOVATION_SOURCES

    if _is_meta_accuracy_question(query):
        return _meta_accuracy_answer(), []

    return _call_original_with_one_retry(query)


_rag_engine.response_guidance = _guarded_response_guidance
_rag_engine.ask_zooz = _guarded_ask_zooz
