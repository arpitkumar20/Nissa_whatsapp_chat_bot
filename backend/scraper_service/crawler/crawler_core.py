from playwright.sync_api import sync_playwright, TimeoutError as PWTimeoutError
from urllib.parse import urlparse, urljoin
from bs4 import BeautifulSoup
import time

class CrawlEngine:
    def __init__(self, start_url, site_id, site_name, max_pages=50, test_mode=True, headless=False, rate_delay=0.8):
        self.start_url = start_url
        self.site_id = site_id
        self.site_name = site_name
        self.max_pages = max_pages
        self.test_mode = test_mode
        self.headless = headless
        self.rate_delay = rate_delay

        self._pw = None
        self._browser = None
        self._context = None
        self._page = None

        self._visited = set()
        self._queue = []
        self._pages_crawled = 0
        self._domain = urlparse(start_url).netloc

    def start(self):
        if self._pw:
            return
        self._pw = sync_playwright().start()
        self._browser = self._pw.chromium.launch(headless=self.headless)
        self._context = self._browser.new_context()
        self._page = self._context.new_page()
        self._page.set_default_navigation_timeout(120000)
        self.enqueue(self.start_url)

    def stop(self):
        try:
            if self._browser:
                self._browser.close()
        except:
            pass
        try:
            if self._pw:
                self._pw.stop()
        except:
            pass
        self._pw = None
        self._browser = None
        self._context = None
        self._page = None

    def enqueue(self, url):
        if url not in self._visited and url not in self._queue:
            self._queue.append(url)

    def same_domain(self, url):
        try:
            return urlparse(url).netloc == self._domain
        except:
            return False

    def fetch(self, url, wait_selector=None, wait_timeout=30000):
        """Navigate the main page to url and return (page, html) or (None, None) when capped."""
        if self._pages_crawled >= self.max_pages:
            return None, None
        try:
            self._page.goto(url, timeout=60000, wait_until="domcontentloaded")
        except PWTimeoutError:
            try:
                self._page.goto(url, timeout=90000)
            except Exception:
                return None, None

        if wait_selector:
            try:
                self._page.wait_for_selector(wait_selector, timeout=wait_timeout)
            except PWTimeoutError:
                pass

        time.sleep(self.rate_delay)
        html = self._page.content()
        self._visited.add(url)
        self._pages_crawled += 1
        return self._page, html

    def visit_detail_and_extract(self, url, extractor_fn, wait_selector=None):
        page, html = self.fetch(url, wait_selector=wait_selector)
        if not page or not html:
            return []
        try:
            recs = extractor_fn(page, html, url)
            return recs or []
        except Exception as e:
            print("Extractor error:", e)
            return []

    def find_links(self, html, base_url):
        soup = BeautifulSoup(html, "html.parser")
        links = []
        for a in soup.select("a[href]"):
            href = a["href"].strip()
            if href.startswith("#") or href.startswith("mailto:") or href.startswith("tel:") or href.startswith("javascript:"):
                continue
            absolute = urljoin(base_url, href)
            if self.same_domain(absolute):
                links.append(absolute.split("#")[0])
        # dedupe preserving order
        seen = set()
        out = []
        for l in links:
            if l not in seen:
                seen.add(l)
                out.append(l)
        return out

    def crawl_discover(self, start_url=None, depth=2, extractor_for_discovered=None):
        if start_url is None:
            start_url = self.start_url
        self.enqueue(start_url)
        results = []
        while self._queue and self._pages_crawled < self.max_pages:
            url = self._queue.pop(0)
            if url in self._visited:
                continue
            page, html = self.fetch(url)
            if not page:
                continue
            if extractor_for_discovered:
                try:
                    recs = extractor_for_discovered(page, html, url)
                    if recs:
                        results.extend(recs)
                except Exception as e:
                    print("discover extractor error:", e)
            links = self.find_links(html, url)
            for l in links:
                if l not in self._visited:
                    self.enqueue(l)
            if self.test_mode and self._pages_crawled >= self.max_pages:
                break
        return results
