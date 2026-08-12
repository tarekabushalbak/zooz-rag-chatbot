import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import json
import time
import os

BASE_URL = "https://www.zooz.co.il"
OUTPUT_FILE = "data/pages.json"
MAX_PAGES = 600

visited = set()
pages_data = []
to_visit = set()

def clean_text(text):
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    return " ".join(lines)

def is_valid_url(url):
    parsed = urlparse(url)
    if parsed.netloc != "www.zooz.co.il":
        return False
    if parsed.scheme not in ["http", "https"]:
        return False
    ext = os.path.splitext(parsed.path)[1].lower()
    if ext in [".jpg", ".jpeg", ".png", ".gif", ".pdf", ".zip", ".css", ".js"]:
        return False
    return True

def get_links(soup, current_url):
    links = set()
    for a in soup.find_all("a", href=True):
        full_url = urljoin(current_url, a["href"])
        full_url = full_url.split("#")[0]
        if is_valid_url(full_url):
            links.add(full_url)
    return links

def scrape_page(url):
    try:
        response = requests.get(url, timeout=10)
        response.encoding = "utf-8"
        soup = BeautifulSoup(response.text, "lxml")
        new_links = get_links(soup, url)
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        title = soup.title.string.strip() if soup.title else ""
        text = clean_text(soup.get_text())
        return {"url": url, "title": title, "text": text}, new_links
    except Exception as e:
        print(f"Error: {url}: {e}")
        return None, set()

def main():
    to_visit.add(BASE_URL)
    to_visit.add("https://www.zooz.co.il/site_map.shtml")
    
    print("Starting recursive crawl...")
    
    while to_visit and len(pages_data) < MAX_PAGES:
        url = to_visit.pop()
        if url in visited:
            continue
        visited.add(url)
        
        print(f"[{len(pages_data)}] Scraping: {url}")
        data, new_links = scrape_page(url)
        
        if data and len(data["text"]) > 100:
            pages_data.append(data)
        
        for link in new_links:
            if link not in visited:
                to_visit.add(link)
        
        time.sleep(0.3)
    
    os.makedirs("data", exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(pages_data, f, ensure_ascii=False, indent=2)
    
    print(f"\nDone! Scraped {len(pages_data)} pages")

if __name__ == "__main__":
    main()
    