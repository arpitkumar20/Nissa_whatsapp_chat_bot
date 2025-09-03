from playwright.sync_api import sync_playwright
from pathlib import Path
from urllib.parse import urljoin, urlparse
import tldextract, os, time
import trafilatura
from robots import RobotsCache

from .ingestion import split_text_to_chunks

class Crawler:
    def __init__(self, out_dir: str):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.robots = RobotsCache()

    def allowed(self, url: str) -> bool:
        try:
            return self.robots.allowed(url, "nisaa-bot")
        except Exception:
            return True

    def normalize_domain_dir(self, base_url: str) -> Path:
        ext = tldextract.extract(base_url)
        domain = ".".join(part for part in [ext.domain, ext.suffix] if part)
        site_dir = self.out_dir / domain
        site_dir.mkdir(parents=True, exist_ok=True)
        return site_dir

    def crawl(self, base_url: str, max_pages: int = 50) -> Path:
        site_dir = self.normalize_domain_dir(base_url)
        visited, to_visit = set(), [base_url]
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            ctx = browser.new_context()
            page = ctx.new_page()
            while to_visit and len(visited) < max_pages:
                url = to_visit.pop(0)
                if url in visited or not self.allowed(url):
                    continue
                try:
                    page.goto(url, timeout=45000, wait_until="domcontentloaded")
                    html = page.content()
                    text = trafilatura.extract(html, favor_recall=True) or ""
                    if len(text.strip()) < 150:
                        continue
                    # save raw text
                    fname = (site_dir / (self.slugify(url) + ".txt"))
                    fname.write_text(text, encoding="utf-8")
                    visited.add(url)

                    # enqueue same-domain links
                    links = page.eval_on_selector_all("a[href]", "els => els.map(e => e.href)")
                    base_host = urlparse(base_url).netloc
                    for link in links:
                        if urlparse(link).netloc == base_host:
                            to_visit.append(link)

                except Exception:
                    continue
            ctx.close(); browser.close()

        # combine to single corpus
        combined = site_dir / "combined.txt"
        with combined.open("w", encoding="utf-8") as f:
            for tfile in site_dir.glob("*.txt"):
                if tfile.name == "combined.txt": continue
                f.write(tfile.read_text(encoding="utf-8").strip() + "\n\n")
        return combined

    @staticmethod
    def slugify(url: str) -> str:
        return url.replace("https://","").replace("http://","").replace("/","_").replace("?","-")[:200]
