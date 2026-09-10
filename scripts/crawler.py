import json
import os
import time
from collections import deque
from urllib.parse import urljoin, urlparse, urldefrag

import requests
from bs4 import BeautifulSoup

BASE_URL = "https://www.zooz.co.il"
OUTPUT_FILE = "data/pages.json"
MAX_PAGES = 600
REQUEST_DELAY_SECONDS = 0.3

visited = set()
pages_data = []
to_visit = deque()


def clean_text(text):
    return " ".join(text.split())


def normalize_url(url):
    url, _ = urldefrag(url)
    parsed = urlparse(url)
    if not parsed.scheme:
        return url
    # Keep the original path/query, but normalize ZOOZ links to HTTPS.
    if parsed.netloc.lower() == "www.zooz.co.il":
        return parsed._replace(scheme="https", netloc="www.zooz.co.il").geturl()
    return url


def is_valid_url(url):
    parsed = urlparse(url)
    if parsed.netloc.lower() != "www.zooz.co.il":
        return False
    if parsed.scheme not in ["http", "https"]:
        return False

    ext = os.path.splitext(parsed.path)[1].lower()
    if ext in [
        ".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg",
        ".pdf", ".zip", ".css", ".js", ".ico", ".xml"
    ]:
        return False
    return True


def get_links(soup, current_url):
    links = set()
    for anchor in soup.find_all("a", href=True):
        full_url = normalize_url(urljoin(current_url, anchor["href"]))
        if is_valid_url(full_url):
            links.add(full_url)
    return sorted(links)


def extract_content_blocks(soup):
    """Extract meaningful page blocks while preserving headings/paragraph boundaries.

    The previous crawler flattened every page into one long line.  Keeping block
    boundaries gives the RAG indexer enough structure to create coherent chunks.
    """
    for tag in soup([
        "script", "style", "nav", "footer", "header", "noscript", "form"
    ]):
        tag.decompose()

    blocks = []
    block_tags = ["h1", "h2", "h3", "h4", "p", "li", "td"]

    for tag in soup.find_all(block_tags):
        # A legacy table cell may contain paragraphs/headings.  In that case we use
        # the child blocks and skip the parent cell to avoid duplicate text.
        if tag.name == "td" and tag.find(["h1", "h2", "h3", "h4", "p", "li"]):
            continue

        text = clean_text(tag.get_text(" ", strip=True))
        if len(text) < 20:
            continue

        block_type = "heading" if tag.name in ["h1", "h2", "h3", "h4"] else "text"
        blocks.append({"type": block_type, "text": text})

    # Remove consecutive duplicates that are common on older HTML pages.
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
            return None, set()

        # Let requests detect older encodings when the server does not declare one
        # correctly.  This is important for Hebrew content on legacy pages.
        if not response.encoding or response.encoding.lower() in ["iso-8859-1", "ascii"]:
            response.encoding = response.apparent_encoding or "utf-8"

        soup = BeautifulSoup(response.text, "lxml")
        new_links = get_links(soup, url)
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
        }, new_links

    except Exception as exc:
        print(f"Error: {url}: {exc}")
        return None, set()


def main():
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (compatible; ZOOZ-RAG-Crawler/2.0)"
    })

    to_visit.append(BASE_URL)
    to_visit.append("https://www.zooz.co.il/site_map.shtml")

    print("Starting structured ZOOZ crawl...")

    queued = set(to_visit)

    while to_visit and len(pages_data) < MAX_PAGES:
        url = to_visit.popleft()
        queued.discard(url)

        if url in visited:
            continue
        visited.add(url)

        print(f"[{len(pages_data)}] Scraping: {url}")
        data, new_links = scrape_page(url, session)

        if data and len(data["text"]) > 100:
            pages_data.append(data)

        for link in new_links:
            if link not in visited and link not in queued:
                to_visit.append(link)
                queued.add(link)

        time.sleep(REQUEST_DELAY_SECONDS)

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        json.dump(pages_data, file, ensure_ascii=False, indent=2)

    print(f"\nDone! Scraped {len(pages_data)} pages")
    print(f"Saved structured content to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
