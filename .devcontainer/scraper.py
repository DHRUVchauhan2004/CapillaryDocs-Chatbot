# scraper.py
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import json
import argparse
import time

HEADERS = {"User-Agent": "Mozilla/5.0 (CapillaryDocBot/1.0)"}

def extract_text_from_soup(soup):
    # Prefer article or main, fallback to p/h tags
    texts = []
    main = soup.find(['article', 'main'])
    if main:
        texts.append(main.get_text(separator="\n"))
    else:
        for tag in soup.find_all(['h1','h2','h3','h4','p','li']):
            texts.append(tag.get_text(separator=" ").strip())
    return "\n".join([t for t in texts if t and len(t.strip())>10])

def crawl(base_url, max_pages=200, delay=0.5):
    parsed = urlparse(base_url)
    domain = parsed.netloc
    scheme = parsed.scheme
    visited = set()
    to_visit = [base_url.rstrip("/")]
    results = []

    while to_visit and len(visited) < max_pages:
        url = to_visit.pop(0)
        if url in visited:
            continue
        try:
            resp = requests.get(url, headers=HEADERS, timeout=15)
            if resp.status_code != 200:
                visited.add(url)
                continue
            soup = BeautifulSoup(resp.text, "html.parser")
            title = soup.title.string.strip() if soup.title else url
            text = extract_text_from_soup(soup)
            if text and len(text) > 20:
                results.append({"url": url, "title": title, "content": text})
                print(f"[+]{len(visited)+1} Saved: {url}")
            visited.add(url)

            # find internal links
            for a in soup.find_all("a", href=True):
                href = a['href'].split('#')[0].strip()
                if not href:
                    continue
                joined = urljoin(url, href)
                p = urlparse(joined)
                # same domain only
                if p.netloc == domain and joined not in visited and joined not in to_visit:
                    if joined.startswith(scheme + "://"):
                        to_visit.append(joined)
            time.sleep(delay)
        except Exception as e:
            print("Error", url, e)
            visited.add(url)
            time.sleep(delay)

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", "-b", required=True, help="Base docs URL, e.g. https://docs.capillarytech.com")
    parser.add_argument("--max", "-m", type=int, default=200, help="Max pages to crawl")
    args = parser.parse_args()

    data = crawl(args.base, max_pages=args.max)
    with open("data.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print("Saved", len(data), "pages to data.json")
