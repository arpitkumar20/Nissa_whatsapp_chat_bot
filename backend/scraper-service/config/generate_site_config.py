import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import re

DEFAULT_TEST_MODE = True
DEFAULT_MAX_PAGES = 50

def slugify_domain(url):
    u = urlparse(url)
    domain = u.netloc.lower().replace("www.", "")
    return re.sub(r'[^a-z0-9\-]', '_', domain)

def detect_namespace_from_text(text, url):
    text = (text or "").lower() + " " + url.lower()
    kws = {
        "hospital": ["hospital","clinic","doctor","appointment","patient","surgery"],
        "hotel": ["hotel","room","suite","booking","reservation"],
        "finance": ["bank","finance","stock","invest"],
        "tech": ["api","developer","software","cloud","product"]
    }
    scores = {}
    for k, arr in kws.items():
        scores[k] = sum(1 for w in arr if w in text)
    best = max(scores.items(), key=lambda x: x[1])
    return best[0] if best[1] > 0 else "general"

def generate_configs(urls):
    configs = {}
    for u in urls:
        try:
            r = requests.get(u, timeout=12, headers={"User-Agent": "general-crawler/1.0"})
            title = ""
            if r.status_code == 200:
                soup = BeautifulSoup(r.text, "html.parser")
                t = soup.title.string.strip() if soup.title and soup.title.string else ""
                title = t or u
            site_id = slugify_domain(u)
            namespace = detect_namespace_from_text(title, u)
            configs[site_id] = {
                "adapter": "crawler.adapters.general_adapter",
                "start_url": u,
                "id": site_id,
                "name": title or site_id,
                "namespace": namespace,
                "test_mode": DEFAULT_TEST_MODE,
                "max_pages": DEFAULT_MAX_PAGES
            }
        except Exception:
            site_id = slugify_domain(u)
            configs[site_id] = {
                "adapter": "crawler.adapters.general_adapter",
                "start_url": u,
                "id": site_id,
                "name": site_id,
                "namespace": "general",
                "test_mode": DEFAULT_TEST_MODE,
                "max_pages": DEFAULT_MAX_PAGES
            }
    return configs
