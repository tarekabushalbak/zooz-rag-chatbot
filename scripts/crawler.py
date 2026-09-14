import json
import os
import time
from collections import deque
from urllib.parse import urljoin, urlparse, urldefrag

import requests
from bs4 import BeautifulSoup

BASE_URL = "https://www.zooz.co.il"
OUTPUT_FILE = "data/pages.json"
PDF_URLS_FILE = "data/pdf_urls.json"
MAX_PAGES = int(os.getenv("ZOOZ_MAX_PAGES", "2500"))
REQUEST_DELAY_SECONDS = 0.3

visited = set()
pages_data = []
pdf_urls = set()
primary_queue = deque()
english_queue = deque()


def clean_text(text):
    return " ".join((text or "").split())


def normalize_url(url):
    url, _ = urldefrag(url)
    parsed = urlparse(url)
    if not parsed.scheme:
        return url

    if parsed.netloc.lower() == "www.zooz.co.il":
        return parsed._replace(scheme="https", netloc="www.zooz.co.il").geturl()
    return url


def is_internal_url(url):
    parsed = urlparse(url)
    return (
        parsed.netloc.lower() == "www.zooz.co.il"
        and parsed.scheme in ["http", "https"]
    )


def path_extension(url):
    return os.path.splitext(urlparse(url).path)[1].lower()


def is_pdf_url(url):
    return is_internal_url(url) and path_extension(url) == ".pdf"


def is_valid_html_url(url):
    if not is_internal_url(url):
        return False

    ext = path_extension(url)
    if ext in [
        ".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg",
        ".pdf", ".zip", ".css", ".js", ".ico", ".xml",
        ".mp3", ".mp4", ".avi", ".mov", ".doc", ".docx",
        ".xls", ".xlsx", ".ppt", ".pptx"
    ]:
        return False
    return True


def is_english_path(url):
    return urlparse(url).path.lower().startswith("/eng/")


def enqueue(url, queued):
    if url in visited or url in queued:
        return
    if is_english_path(url):
        english_queue.append(url)
    else:
        primary_queue.append(url)
    queued.add(url)


def get_links(soup, current_url):
    html_links = set()
    discovered_pdfs = set()

    for anchor in soup.find_all("a", href=True):
        full_url = normalize_url(urljoin(current_url, anchor["href"]))
        if is_pdf_url(full_url):
            discovered_pdfs.add(full_url)
        elif is_valid_html_url(full_url):
            html_links.add(full_url)

    # Deterministic ordering makes repeated crawls easier to compare.
    return sorted(html_links), sorted(discovered_pdfs)


def extract_content_blocks(soup):
    """Extract useful text while preserving headings and paragraph boundaries."""
    for tag in soup([
        "script", "style", "nav", "footer", "header", "noscript", "form"
    ]):
        tag.decompose()

    blocks = []
    block_tags = ["h1", "h2", "h3", "h4", "p", "li", "td"]

    for tag in soup.find_all(block_tags):
        # Legacy ZOOZ pages often use tables as layout containers. If a table cell
        # already contains semantic child blocks, keep the children and skip the
        # parent cell to avoid indexing the same content twice.
        if tag.name == "td" and tag.find(["h1", "h2", "h3", "h4", "p", "li"]):
            continue

        text = clean_text(tag.get_text(" ", strip=True))
        if len(text) < 20:
            continue

        block_type = "heading" if tag.name in ["h1", "h2", "h3", "h4"] else "text"
        blocks.append({"type": block_type, "text": text})

    deduped = []
    previous = None
    for block in blocks:
        if block["text"] == previous:
            continue
        deduped.append(block)
        previous = block["text"]

    return deduped


def scrape_page(url, session):
    try:
        response = session.get(url, timeout=15)
        response.raise_for_status()

        content_type = response.headers.get("content-type", "").lower()
        if content_type and "html" not in content_type:
            return None, [], []

        # Many ZOOZ pages are legacy HTML. Prefer the server declaration, but when
        # it falls back to a generic encoding let requests inspect the bytes.
        if not response.encoding or response.encoding.lower() in ["iso-8859-1", "ascii"]:
            response.encoding = response.apparent_encoding or "utf-8"

        soup = BeautifulSoup(response.text, "lxml")
        new_links, new_pdfs = get_links(soup, url)
        title = clean_text(soup.title.get_text(" ", strip=True)) if soup.title else ""
        blocks = extract_content_blocks(soup)

        if blocks:
            text = "\n".join(block["text"] for block in blocks)
        else:
            text = clean_text(soup.get_text(" ", strip=True))

        return {
            "url": url,
            "title": title,
            "text": text,
            "blocks": blocks,
            "source_type": "html",
        }, new_links, new_pdfs

    except Exception as exc:
        print(f"Error: {url}: {exc}")
        return None, [], []


def main():
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (compatible; ZOOZ-RAG-Crawler/2.1)"
    })

    queued = set()
    enqueue(BASE_URL, queued)
    enqueue("https://www.zooz.co.il/site_map.shtml", queued)

    print("Starting structured ZOOZ crawl...")
    print(f"Safety limit: {MAX_PAGES} indexed HTML pages")
    print("Hebrew/non-/eng/ URLs are crawled before /eng/ URLs.")

    while (primary_queue or english_queue) and len(pages_data) < MAX_PAGES:
        queue = primary_queue if primary_queue else english_queue
        url = queue.popleft()
        queued.discard(url)

        if url in visited:
            continue
        visited.add(url)

        print(f"[{len(pages_data)}] Scraping: {url}")
        data, new_links, new_pdfs = scrape_page(url, session)
        pdf_urls.update(new_pdfs)

        if data and len(data["text"]) > 100:
            pages_data.append(data)

        for link in new_links:
            enqueue(link, queued)

        time.sleep(REQUEST_DELAY_SECONDS)

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        json.dump(pages_data, file, ensure_ascii=False, indent=2)

    with open(PDF_URLS_FILE, "w", encoding="utf-8") as file:
        json.dump(sorted(pdf_urls), file, ensure_ascii=False, indent=2)

    print(f"\nDone! Scraped {len(pages_data)} HTML pages")
    print(f"Discovered {len(pdf_urls)} PDF links")
    print(f"Saved structured content to {OUTPUT_FILE}")
    print(f"Saved PDF inventory to {PDF_URLS_FILE}")

    if len(pages_data) >= MAX_PAGES and (primary_queue or english_queue):
        print("WARNING: crawl hit the safety limit while URLs still remained in the queue.")
        print("Increase ZOOZ_MAX_PAGES and run again before building ChromaDB.")
    else:
        print("Crawl queue exhausted before the safety limit: HTML coverage is complete for discovered links.")


if __name__ == "__main__":
    main()
