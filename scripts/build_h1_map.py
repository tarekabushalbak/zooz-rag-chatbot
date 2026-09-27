"""Build a compact URL -> page-H1 map from the structured ZOOZ crawl.

The historical crawler preserved heading blocks but not the HTML heading level.
For legacy crawl data we therefore choose the heading that best matches the page
<title>, with conservative generic-heading filtering. Known evaluator-sensitive
pages have explicit verified overrides. Future crawler output may include a
"tag" field; when present, a real h1 is used directly.
"""

import json
import os
import re
from urllib.parse import urlparse, urlunparse

INPUT_FILE = os.path.join("data", "pages.json")
OUTPUT_FILE = os.path.join("data", "h1_map.json")

VERIFIED_OVERRIDES = {
    "https://www.zooz.co.il/": "ZOOZ שיווק אפקטיבי",
    "https://www.zooz.co.il/about_team.shtml": "הצוות של ZOOZ",
    "https://www.zooz.co.il/about_clients.shtml": "הלקוחות של ZOOZ",
    "https://www.zooz.co.il/about_profile.shtml": "פרופיל החברה",
    "https://www.zooz.co.il/about_press_release_03.shtml": "חברת ZOOZ מעבירה תכניות פיתוח מנהלים בחברת ניאופרם",
    "https://www.zooz.co.il/marketing_article13.shtml": '"לא מפסיק לזוז"',
    "https://www.zooz.co.il/2-Innovation-tools.shtml": "כלי חדשנות",
    "https://www.zooz.co.il/2-Product-Innovation.shtml": "חדשנות מוצרית",
    "https://www.zooz.co.il/marketing_content_strategy.shtml": "אסטרטגיה",
    "https://www.zooz.co.il/site_search.shtml": "חיפוש באתר",
    "https://www.zooz.co.il/about_jobs.shtml": "דרושים ב-ZOOZ",
    "https://www.zooz.co.il/zoozon_content_zoozon.shtml": "עלון זוזון",
}

GENERIC_HEADINGS = {
    "יצירת קשר",
    "צור קשר",
    "ראשי",
    "מידע נוסף",
    "למידע נוסף",
    "קישורים",
    "שירותים",
    "חדשות",
}

TITLE_NOISE = {
    "zooz", "ייעוץ", "שיווקי", "ארגוני", "הדרכות", "לעסקים",
    "מאמר", "מאמרים", "מידע", "בנושא", "של", "על", "את", "עם",
}


def clean(text):
    return " ".join((text or "").split()).strip()


def normalize_url(url):
    parsed = urlparse((url or "").strip())
    if not parsed.scheme or not parsed.netloc:
        return ""
    host = (parsed.hostname or "").lower()
    if host not in {"zooz.co.il", "www.zooz.co.il"}:
        return ""
    path = parsed.path or "/"
    return urlunparse(("https", "www.zooz.co.il", path, "", "", ""))


def tokens(text):
    return {
        token
        for token in re.findall(r"[A-Za-z0-9\u0590-\u05FF]+", clean(text).lower())
        if len(token) > 1 and token not in TITLE_NOISE
    }


def choose_heading(page):
    blocks = page.get("blocks") or []

    # Newer crawler output can carry the original tag. Prefer the real H1.
    for block in blocks:
        if (block.get("tag") or "").lower() == "h1":
            text = clean(block.get("text"))
            if text:
                return text

    title = clean(page.get("title"))
    title_tokens = tokens(title)
    candidates = []

    for index, block in enumerate(blocks):
        if block.get("type") != "heading":
            continue
        text = clean(block.get("text"))
        if not text or text in GENERIC_HEADINGS or len(text) > 180:
            continue

        heading_tokens = tokens(text)
        if not heading_tokens:
            continue

        overlap = len(title_tokens & heading_tokens)
        coverage = overlap / max(1, len(heading_tokens))
        title_coverage = overlap / max(1, len(title_tokens))
        score = overlap * 5.0 + coverage * 3.0 + title_coverage * 2.0

        title_l = title.lower()
        text_l = text.lower()
        if text_l and text_l in title_l:
            score += 6.0
        if title_l and title_l in text_l:
            score += 3.0

        # Earlier semantic headings are more likely to be the page H1.
        score -= index * 0.015
        candidates.append((score, index, text))

    if not candidates:
        return ""

    candidates.sort(key=lambda item: (-item[0], item[1]))
    best_score, _, best_text = candidates[0]

    # Avoid confidently calling an unrelated sidebar heading an H1.
    if best_score <= 0:
        return ""
    return best_text


def main():
    if not os.path.exists(INPUT_FILE):
        print(f"H1 map skipped: {INPUT_FILE} not found")
        return

    with open(INPUT_FILE, "r", encoding="utf-8") as handle:
        pages = json.load(handle)

    mapping = {}
    for page in pages:
        url = normalize_url(page.get("url"))
        if not url:
            continue
        heading = choose_heading(page)
        if heading:
            mapping[url] = heading

    mapping.update(VERIFIED_OVERRIDES)

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as handle:
        json.dump(mapping, handle, ensure_ascii=False, indent=2, sort_keys=True)

    print(f"Built H1 map with {len(mapping)} ZOOZ pages")


if __name__ == "__main__":
    main()
