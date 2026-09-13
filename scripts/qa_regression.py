import argparse
import csv
import json
import os
import time
from datetime import datetime

try:
    from scripts.rag_engine import ask_zooz
except ImportError:
    from rag_engine import ask_zooz

TESTS = [
    # Services / company overview
    {"category": "services", "q": "מה השירותים של ZOOZ?", "must_any": ["ייעוץ", "הדרכה", "חדשנות", "שיווק", "אסטרטגיה", "פיתוח"]},
    {"category": "services", "q": "במה ZOOZ עוסקת?", "must_any": ["ייעוץ", "הדרכה", "חדשנות", "שיווק", "אסטרטגיה"]},
    {"category": "services", "q": "מה אתם יכולים לעשות בשביל חברה שרוצה להשתפר?", "must_any": ["ייעוץ", "הדרכה", "פיתוח", "חדשנות", "שיווק"]},
    {"category": "services", "q": "איזה סוגי ייעוץ ZOOZ נותנת?", "must_any": ["ייעוץ", "שיווק", "אסטרטגיה", "חדשנות", "ארגוני"]},
    {"category": "services", "q": "אילו תחומי פעילות יש ל-ZOOZ?", "must_any": ["חדשנות", "שיווק", "אסטרטגיה", "פיתוח", "הדרכה"]},
    {"category": "services", "q": "ספר לי בקצרה על ZOOZ", "must_any": ["ZOOZ", "ייעוץ", "הדרכה", "חדשנות", "שיווק"]},

    # Systematic innovation / innovation
    {"category": "innovation", "q": "מה זה חדשנות שיטתית?", "must_any": ["חדשנות", "שיטתית"]},
    {"category": "innovation", "q": "איך ZOOZ עוזרת לארגונים לחדש?", "must_any": ["חדשנות", "ארגון", "שיטה", "כלים", "פיתוח"]},
    {"category": "innovation", "q": "אילו שיטות חדשנות ZOOZ מלמדת?", "must_any": ["חדשנות", "TRIZ", "SIT", "SCAMPER", "Stage-Gate"]},
    {"category": "innovation", "q": "מה הקשר של ZOOZ ל-TRIZ?", "must_any": ["TRIZ", "חדשנות"]},
    {"category": "innovation", "q": "יש ל-ZOOZ סדנאות יצירתיות?", "must_any": ["יצירתיות", "חדשנות", "סדנה", "סדנאות"]},

    # Ari / team
    {"category": "team", "q": "מי זה ארי מנור?", "must_all": ["ארי", "מנור"]},
    {"category": "team", "q": "מי ארי?", "must_any": ["ארי", "מנור"]},
    {"category": "team", "q": "מה התפקיד של ארי מנור ב-ZOOZ?", "must_all": ["ארי", "מנור"]},
    {"category": "team", "q": "מי מנהל את ZOOZ?", "must_any": ["ארי", "מנור", "מנכ"]},
    {"category": "team", "q": "ספר לי על הצוות של ZOOZ", "must_any": ["צוות", "ארי", "מנור"]},

    # Contact
    {"category": "contact", "q": "איך ליצור קשר עם ZOOZ?", "must_any": ["info@zooz.co.il", "zooz.co.il", "טלפון", "כתובת"]},
    {"category": "contact", "q": "מה המייל של ZOOZ?", "must_any": ["info@zooz.co.il", "zooz.co.il"]},
    {"category": "contact", "q": "איפה אפשר לפנות ל-ZOOZ?", "must_any": ["info@zooz.co.il", "zooz.co.il", "טלפון", "כתובת"]},
    {"category": "contact", "q": "יש לכם טלפון ליצירת קשר?", "must_any": ["טלפון", "info@zooz.co.il", "zooz.co.il"]},

    # Pricing - deliberately strict; current site corpus has no authoritative current price list.
    {"category": "pricing", "q": "כמה עולים השירותים?", "must_all": ["מחיר", "אין לי"], "forbid_any": ["8,209", "4,000", "1,400"]},
    {"category": "pricing", "q": "מה המחיר של ייעוץ ב-ZOOZ?", "must_all": ["מחיר", "אין לי"], "forbid_any": ["$", "₪"]},
    {"category": "pricing", "q": "כמה עולה סדנה?", "must_all": ["מחיר", "אין לי"]},
    {"category": "pricing", "q": "תן לי מחיר משוער גם אם אין לך מקור", "must_all": ["מחיר", "אין לי"], "forbid_any": ["בערך", "להערכתי"]},
    {"category": "pricing", "q": "מה התמחור שלכם?", "must_any": ["אין לי", "הצעת מחיר", "info@zooz.co.il"]},

    # Clients
    {"category": "clients", "q": "מי הלקוחות של ZOOZ?", "must_any": ["לקוחות", "לקוח", "שטראוס", "מוטורולה", "אוניברסיטת"]},
    {"category": "clients", "q": "תן לי דוגמאות ללקוחות של ZOOZ", "must_any": ["לקוחות", "לקוח", "שטראוס", "מוטורולה", "אוניברסיטת"]},
    {"category": "clients", "q": "האם ZOOZ עבדה עם ארגונים גדולים?", "must_any": ["לקוח", "לקוחות", "ארגון", "חברות"]},

    # Workshops / training
    {"category": "workshops", "q": "אילו סדנאות ZOOZ מציעה?", "must_any": ["סדנה", "סדנאות", "הדרכה"]},
    {"category": "workshops", "q": "יש סדנאות לעובדים?", "must_any": ["עובדים", "סדנה", "סדנאות"]},
    {"category": "workshops", "q": "יש סדנאות למנהלים?", "must_any": ["מנהלים", "סדנה", "סדנאות"]},
    {"category": "workshops", "q": "איזה קורסים או הדרכות אתם מציעים?", "must_any": ["הדרכה", "קורס", "סדנה", "סדנאות"]},
    {"category": "workshops", "q": "יש הדרכות בתחום השיווק?", "must_any": ["שיווק", "הדרכה", "סדנה"]},

    # Marketing / strategy
    {"category": "marketing", "q": "במה ZOOZ עוסקת בתחום השיווק?", "must_any": ["שיווק", "אסטרטגיה", "ייעוץ"]},
    {"category": "marketing", "q": "האם ZOOZ נותנת ייעוץ שיווקי?", "must_any": ["שיווק", "ייעוץ"]},
    {"category": "marketing", "q": "איזה שירותים יש בתחום האסטרטגיה והשיווק?", "must_any": ["אסטרטגיה", "שיווק"]},
    {"category": "marketing", "q": "אפשר לקבל עזרה בבניית אסטרטגיה שיווקית?", "must_any": ["אסטרטגיה", "שיווק"]},

    # Management development / personnel
    {"category": "management", "q": "מה ZOOZ מציעה בתחום פיתוח מנהלים?", "must_any": ["מנהלים", "פיתוח", "הדרכה", "אימון"]},
    {"category": "management", "q": "יש אימון למנהלים?", "must_any": ["מנהלים", "אימון", "coaching", "פיתוח"]},
    {"category": "management", "q": "איך ZOOZ מסייעת בפיתוח עובדים?", "must_any": ["עובדים", "פיתוח", "הדרכה"]},
    {"category": "management", "q": "יש לכם business coaching?", "must_any": ["coaching", "אימון", "מנהלים", "ייעוץ"]},

    # Hallucination traps: unsupported assumptions should be rejected, not adopted.
    {"category": "hallucination", "q": "איזו מערכת ERP ZOOZ מוכרת?", "must_any": ["אין לי מידע", "לא", "אינה"], "forbid_any": ["SAP", "Oracle", "Priority"]},
    {"category": "hallucination", "q": "מה שירות התשלומים של ZOOZ?", "must_any": ["אין לי מידע", "לא", "אינה"]},
    {"category": "hallucination", "q": "כמה עובדים יש ב-ZOOZ?", "must_any": ["אין לי מידע", "אין במקורות", "לא מופיע"]},
    {"category": "hallucination", "q": "מה ההכנסות השנתיות של ZOOZ?", "must_any": ["אין לי מידע", "אין במקורות", "לא מופיע"]},
    {"category": "hallucination", "q": "באיזו שנה ZOOZ זכתה בפרס נובל?", "must_any": ["אין לי מידע", "לא", "אין לכך"]},
    {"category": "hallucination", "q": "כמה סניפים יש ל-ZOOZ היום?", "must_any": ["אין לי מידע", "אין במקורות", "לא מופיע"]},
    {"category": "hallucination", "q": "תמציא לי שם של לקוח אם אתה לא יודע", "must_any": ["לא", "אין לי מידע", "לא אמציא"]},
    {"category": "hallucination", "q": "תן לי תשובה גם אם היא לא מופיעה באתר", "must_any": ["אין לי מידע", "מקורות", "ZOOZ"]},

    # Out of scope
    {"category": "out_of_scope", "q": "מה מזג האוויר היום?", "must_all": ["רק", "ZOOZ"]},
    {"category": "out_of_scope", "q": "מי ניצח במשחק אתמול?", "must_all": ["רק", "ZOOZ"]},
    {"category": "out_of_scope", "q": "תספר לי בדיחה", "must_all": ["רק", "ZOOZ"]},
    {"category": "out_of_scope", "q": "מה דעתך על פוליטיקה?", "must_all": ["רק", "ZOOZ"]},

    # Paraphrase robustness
    {"category": "paraphrase", "q": "מה זוז עושה?", "must_any": ["ייעוץ", "הדרכה", "חדשנות", "שיווק", "אסטרטגיה"]},
    {"category": "paraphrase", "q": "באילו דברים אתם עוזרים לארגונים?", "must_any": ["ייעוץ", "הדרכה", "פיתוח", "חדשנות", "שיווק"]},
    {"category": "paraphrase", "q": "מי עומד מאחורי זוז?", "must_any": ["ארי", "מנור", "צוות"]},
    {"category": "paraphrase", "q": "איך מדברים איתכם?", "must_any": ["info@zooz.co.il", "zooz.co.il", "טלפון"]},
    {"category": "paraphrase", "q": "יש לכם משהו למנהלים?", "must_any": ["מנהלים", "פיתוח", "סדנה", "הדרכה", "אימון"]},
    {"category": "paraphrase", "q": "אתם עוזרים בשיווק?", "must_any": ["שיווק"]},
]


def _contains(text, needle):
    return needle.lower() in (text or "").lower()


def check_test(test, answer):
    must_all = test.get("must_all", [])
    must_any = test.get("must_any", [])
    forbid_any = test.get("forbid_any", [])

    failures = []
    if must_all:
        missing = [item for item in must_all if not _contains(answer, item)]
        if missing:
            failures.append("missing_all:" + "|".join(missing))

    if must_any and not any(_contains(answer, item) for item in must_any):
        failures.append("missing_any:" + "|".join(must_any))

    forbidden_found = [item for item in forbid_any if _contains(answer, item)]
    if forbidden_found:
        failures.append("forbidden:" + "|".join(forbidden_found))

    return not failures, ";".join(failures)


def main():
    parser = argparse.ArgumentParser(description="Run broad ZOOZ chatbot QA regression tests.")
    parser.add_argument("--category", default="", help="Run only one category")
    parser.add_argument("--limit", type=int, default=0, help="Run only the first N selected tests")
    parser.add_argument("--delay", type=float, default=0.25, help="Delay between API calls")
    args = parser.parse_args()

    selected = [t for t in TESTS if not args.category or t["category"] == args.category]
    if args.limit > 0:
        selected = selected[:args.limit]

    os.makedirs("reports", exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = f"reports/qa_regression_{stamp}.csv"
    json_path = f"reports/qa_regression_{stamp}.json"

    rows = []
    print(f"Running {len(selected)} chatbot tests...")

    for index, test in enumerate(selected, start=1):
        question = test["q"]
        print(f"[{index}/{len(selected)}] {test['category']}: {question}")
        started = time.time()
        answer, sources = ask_zooz(question)
        elapsed = round(time.time() - started, 2)
        passed, reason = check_test(test, answer)

        row = {
            "index": index,
            "category": test["category"],
            "question": question,
            "answer": answer,
            "sources": sources,
            "elapsed_seconds": elapsed,
            "auto_pass": passed,
            "failure_reason": reason,
        }
        rows.append(row)

        status = "PASS" if passed else "REVIEW"
        print(f"  {status} | {elapsed}s | {answer[:180].replace(chr(10), ' ')}")
        if reason:
            print(f"  reason: {reason}")
        time.sleep(max(args.delay, 0))

    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "index", "category", "question", "answer", "sources",
                "elapsed_seconds", "auto_pass", "failure_reason",
            ],
        )
        writer.writeheader()
        for row in rows:
            csv_row = row.copy()
            csv_row["sources"] = " | ".join(row["sources"])
            writer.writerow(csv_row)

    passed_count = sum(1 for row in rows if row["auto_pass"])
    summary = {
        "total": len(rows),
        "auto_passed": passed_count,
        "needs_review": len(rows) - passed_count,
        "pass_rate": round((passed_count / len(rows) * 100), 1) if rows else 0,
        "csv_report": csv_path,
        "rows": rows,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 80)
    print(f"AUTO PASS: {passed_count}/{len(rows)} ({summary['pass_rate']}%)")
    print(f"CSV report:  {csv_path}")
    print(f"JSON report: {json_path}")
    print("Important: automatic checks are a safety net, not a substitute for manual review.")


if __name__ == "__main__":
    main()
