import json
import re
import hashlib
import os
from collections import Counter
from urllib.parse import urlparse

INPUT_FILE = "data/pages.json"
PDF_INVENTORY_FILE = "data/pdf_urls.json"
CRAWL_SAFETY_LIMIT = 2500

HEBREW_RE = re.compile(r"[\u0590-\u05FF]")
LATIN_RE = re.compile(r"[A-Za-z]")
MOJIBAKE_MARKERS = ("�", "Ã", "Â", "×", "Ø", "Ù")

KEYWORDS = [
    "חדשנות",
    "חדשנות שיטתית",
    "ארי מנור",
    "אסטרטגיה",
    "שיווק",
    "פיתוח מנהלים",
    "סדנאות",
    "יצירת קשר",
    "לקוחות",
]


def normalize(text):
    return " ".join((text or "").split())


def text_language(text):
    heb = len(HEBREW_RE.findall(text or ""))
    lat = len(LATIN_RE.findall(text or ""))
    if heb > lat * 1.5:
        return "hebrew-dominant"
    if lat > heb * 1.5:
        return "latin-dominant"
    return "mixed"


def path_bucket(url):
    path = urlparse(url).path.strip("/")
    if not path:
        return "/"
    parts = path.split("/")
    return "/" + "/".join(parts[:2])


def load_pdf_count():
    if not os.path.exists(PDF_INVENTORY_FILE):
        return 0
    try:
        with open(PDF_INVENTORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return len(data) if isinstance(data, list) else 0
    except Exception:
        return 0


def main():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        pages = json.load(f)

    total = len(pages)
    urls = [p.get("url", "") for p in pages]
    texts = [normalize(p.get("text", "")) for p in pages]

    url_counts = Counter(urls)
    duplicate_urls = [u for u, c in url_counts.items() if u and c > 1]

    hashes = Counter(
        hashlib.sha1(t.encode("utf-8")).hexdigest()
        for t in texts if t
    )
    duplicate_content_groups = sum(1 for c in hashes.values() if c > 1)

    languages = Counter(text_language(t) for t in texts)
    eng_path_count = sum(1 for u in urls if "/eng/" in u.lower())
    structured_count = sum(1 for p in pages if p.get("blocks"))
    mojibake_pages = [
        p.get("url", "") for p in pages
        if any(marker in (p.get("text", "") or "") for marker in MOJIBAKE_MARKERS)
    ]

    lengths = [len(t) for t in texts if t]
    avg_len = round(sum(lengths) / len(lengths), 1) if lengths else 0
    min_len = min(lengths) if lengths else 0
    max_len = max(lengths) if lengths else 0
    pdf_count = load_pdf_count()

    print("=" * 80)
    print("ZOOZ CRAWL AUDIT")
    print("=" * 80)
    print(f"Total pages:                 {total}")
    print(f"Structured pages (blocks):   {structured_count}")
    print(f"English-path pages (/eng/):  {eng_path_count}")
    print(f"Hebrew-dominant pages:        {languages['hebrew-dominant']}")
    print(f"Latin-dominant pages:         {languages['latin-dominant']}")
    print(f"Mixed pages:                  {languages['mixed']}")
    print(f"Duplicate URL count:          {len(duplicate_urls)}")
    print(f"Duplicate content groups:     {duplicate_content_groups}")
    print(f"Possible mojibake pages:      {len(mojibake_pages)}")
    print(f"Text length avg/min/max:      {avg_len} / {min_len} / {max_len}")
    print(f"Discovered PDF links:         {pdf_count}")

    print("\nTop path buckets:")
    for bucket, count in Counter(path_bucket(u) for u in urls if u).most_common(15):
        print(f"  {count:4d}  {bucket}")

    print("\nKeyword coverage (pages containing term):")
    for keyword in KEYWORDS:
        count = sum(1 for t in texts if keyword.lower() in t.lower())
        print(f"  {count:4d}  {keyword}")

    if mojibake_pages:
        print("\nFirst possible encoding problems:")
        for url in mojibake_pages[:10]:
            print(f"  {url}")

    print("\nLast 15 crawled URLs:")
    for url in urls[-15:]:
        print(f"  {url}")

    if total >= CRAWL_SAFETY_LIMIT:
        print(f"\nWARNING: crawl reached the safety limit ({CRAWL_SAFETY_LIMIT} pages).")
        print("The site may have been truncated before all discovered HTML pages were crawled.")
        print("Do NOT rebuild ChromaDB until this is reviewed.")
    else:
        print(f"\nOK: crawl stayed below the safety limit ({CRAWL_SAFETY_LIMIT}).")
        print("If the crawler also reported 'Crawl queue exhausted', discovered HTML coverage is complete.")


if __name__ == "__main__":
    main()
