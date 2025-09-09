import re
import time
import json
import hashlib
from urllib.parse import urlparse, urljoin
from typing import List, Dict, Any
from bs4 import BeautifulSoup
import requests
from playwright.sync_api import sync_playwright, TimeoutError as PWTimeoutError

try:
    from readability import Document as ReadabilityDocument
    _HAS_READABILITY = True
except Exception:
    ReadabilityDocument = None
    _HAS_READABILITY = False

# ---------------- Config ----------------
CHUNK_TARGET = 900
CHUNK_OVERLAP = 200
MAX_CHUNKS_PER_PAGE = 500
MAX_CHARS_PER_PAGE = 2500000
MAX_ENQUEUE_PER_PAGE = 2000
PER_SECTION_LIMIT = 300
DEFAULT_USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120 Safari/537.36"
PAGE_GOTO_TIMEOUT_MS = 4500000
PAGE_WAIT_AFTER_LOAD = 3.0  

# ---------------- Utilities ----------------
def _short_hash(s: str, n: int = 12) -> str:
    return hashlib.sha1((s or "").encode("utf-8")).hexdigest()[:n]

def _make_chunk_id(site_id: str, page_id: str, idx: int) -> str:
    return f"{site_id}_page_{page_id}_{idx}"

def _slugify_domain(url: str) -> str:
    try:
        u = urlparse(url)
        domain = u.netloc.lower().replace("www.", "")
        return re.sub(r'[^a-z0-9\-]', '_', domain)
    except Exception:
        return "site"

_MAX_META_STR = 16000
def _sanitize_value(v):
    if v is None:
        return None
    if isinstance(v, (str, int, float, bool)):
        s = v if not isinstance(v, str) else (v if len(v) <= _MAX_META_STR else v[:_MAX_META_STR])
        return s
    if isinstance(v, list):
        if all(isinstance(x, (str, int, float, bool)) for x in v):
            return [x if not isinstance(x, str) or len(x) <= _MAX_META_STR else x[:_MAX_META_STR] for x in v]
        try:
            s = json.dumps(v, ensure_ascii=False)
            return s[:_MAX_META_STR]
        except Exception:
            return str(v)[:_MAX_META_STR]
    try:
        s = json.dumps(v, ensure_ascii=False)
        return s[:_MAX_META_STR]
    except Exception:
        return str(v)[:_MAX_META_STR]

def sanitize_metadata(meta: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    if not meta:
        return out
    for k, v in meta.items():
        sv = _sanitize_value(v)
        if sv is not None:
            out[k] = sv
    return out

def _collapse_spaces(text: str) -> str:
    return re.sub(r'\s+', ' ', (text or "")).strip()

# ---------------- Structured data extraction ----------------
def _extract_json_ld(soup: BeautifulSoup) -> List[Dict[str, Any]]:
    out = []
    for sc in soup.select("script[type='application/ld+json']"):
        try:
            txt = sc.string or sc.get_text()
            if not txt or txt.strip() == "":
                continue
            payload = json.loads(txt)
            # payload can be list or dict - normalize to list
            if isinstance(payload, list):
                out.extend(payload)
            else:
                out.append(payload)
        except Exception:
            # try to recover multiple JSON objects in same script
            try:
                txt = sc.string or sc.get_text() or ""
                parts = re.split(r'\}\s*\{', txt)
                if len(parts) > 1:
                    fixed = []
                    for i, p in enumerate(parts):
                        if i == 0:
                            fixed.append(p + "}")
                        elif i == len(parts) - 1:
                            fixed.append("{" + p)
                        else:
                            fixed.append("{" + p + "}")
                    for f in fixed:
                        try:
                            out.append(json.loads(f))
                        except Exception:
                            continue
            except Exception:
                continue
    return out

# ---------------- Generic extractors ----------------
def _extract_tables(soup: BeautifulSoup, url: str, site_id: str) -> List[Dict[str, Any]]:
    out = []
    tables = soup.select("table")
    for i, t in enumerate(tables):
        try:
            text = []
            headers = [th.get_text(" ", strip=True) for th in t.select("thead th")] or []
            if headers:
                text.append(" | ".join(headers))
            for tr in t.select("tbody tr") or t.select("tr"):
                cols = [td.get_text(" ", strip=True) for td in tr.select("td, th")]
                if cols:
                    text.append(" | ".join(cols))
            if not text:
                continue
            tid = f"{site_id}_table_{_short_hash(url + str(i))}"
            meta = {"site_id": site_id, "entity_type": "table", "entity_id": tid, "source_url": url}
            out.append({"id": tid, "text": _collapse_spaces("\n".join(text)), "metadata": sanitize_metadata(meta)})
        except Exception:
            continue
    return out

def _extract_definition_lists(soup: BeautifulSoup, url: str, site_id: str) -> List[Dict[str, Any]]:
    out = []
    dls = soup.select("dl")
    for i, dl in enumerate(dls):
        try:
            items = []
            for dt, dd in zip(dl.select("dt"), dl.select("dd")):
                dt_text = dt.get_text(" ", strip=True)
                dd_text = dd.get_text(" ", strip=True)
                items.append(f"{dt_text}: {dd_text}")
            if items:
                did = f"{site_id}_dl_{_short_hash(url + str(i))}"
                meta = {"site_id": site_id, "entity_type": "definition_list", "entity_id": did, "source_url": url}
                out.append({"id": did, "text": _collapse_spaces(" • ".join(items)), "metadata": sanitize_metadata(meta)})
        except Exception:
            continue
    return out

def _extract_list_cards(soup: BeautifulSoup, url: str, site_id: str) -> List[Dict[str, Any]]:
    out = []
    candidates = soup.find_all(lambda tag: tag.name in ("div", "section", "ul", "ol")
                               and len(tag.find_all(recursive=False)) >= 2)
    seen = set()
    for c in candidates:
        children = [ch for ch in c.find_all(recursive=False) if getattr(ch, "get_text", None)]
        if len(children) < 2 or len(children) > 200:
            continue
        avg_len = sum(len(_collapse_spaces(ch.get_text(" ", strip=True))) for ch in children) / max(1, len(children))
        if avg_len < 30:
            continue
        for i, ch in enumerate(children[:PER_SECTION_LIMIT]):
            try:
                txt = _collapse_spaces(ch.get_text(" ", strip=True))
                if not txt:
                    continue
                cid = f"{site_id}_card_{_short_hash(url + str(hash(ch)) + str(i))}"
                meta = {"site_id": site_id, "entity_type": "card", "entity_id": cid, "source_url": url}
                if cid not in seen:
                    out.append({"id": cid, "text": txt, "metadata": sanitize_metadata(meta)})
                    seen.add(cid)
            except Exception:
                continue
        if out:
            break
    return out

# ---------------- Chunking helpers ----------------
def chunk_by_paragraphs(text: str, site_id: str, page_id: str, target: int = CHUNK_TARGET) -> List[Dict[str, Any]]:
    if not text:
        return []
    text = re.sub(r'\s+', ' ', text).strip()
    if len(text) > MAX_CHARS_PER_PAGE:
        text = text[:MAX_CHARS_PER_PAGE]
    paragraphs = [p.strip() for p in re.split(r'\n{1,}|\r{1,}|(?<=\.)\s{2,}', text) if p.strip()]
    if not paragraphs:
        paragraphs = [text]
    chunks = []
    cur = ""
    idx = 0
    for p in paragraphs:
        if len(cur) + len(p) + 1 <= target:
            cur = (cur + " " + p).strip()
        else:
            if cur:
                chunks.append({"id": _make_chunk_id(site_id, page_id, idx), "text": cur})
                idx += 1
            if len(p) > target:
                for i in range(0, len(p), target - CHUNK_OVERLAP):
                    if idx >= MAX_CHUNKS_PER_PAGE:
                        break
                    chunks.append({"id": _make_chunk_id(site_id, page_id, idx), "text": p[i:i+target]})
                    idx += 1
                cur = ""
            else:
                cur = p
        if idx >= MAX_CHUNKS_PER_PAGE:
            break
    if cur and idx < MAX_CHUNKS_PER_PAGE:
        chunks.append({"id": _make_chunk_id(site_id, page_id, idx), "text": cur})
    if not chunks:
        L = len(text)
        i = 0
        idx = 0
        while i < L and idx < MAX_CHUNKS_PER_PAGE:
            end = min(i + target, L)
            chunks.append({"id": _make_chunk_id(site_id, page_id, idx), "text": text[i:end]})
            i = max(end - CHUNK_OVERLAP, end - target)
            idx += 1
    return chunks[:MAX_CHUNKS_PER_PAGE]

# ---------------- Page section extractor (headings) ----------------
def _extract_sections_by_headings(soup: BeautifulSoup) -> List[Dict[str, Any]]:

    body = soup.body or soup
    sections = []
    current_heading = ""
    current_text = []
    for el in body.descendants:
        if not getattr(el, "name", None):
            continue
        if el.name in ("h1", "h2", "h3", "h4", "h5", "h6"):
            if current_text:
                sections.append({"heading": current_heading, "text": _collapse_spaces(" ".join(current_text))})
                current_text = []
            current_heading = _collapse_spaces(el.get_text(" ", strip=True)) or current_heading
            continue
        if el.name in ("p", "div", "section", "article", "li"):
            txt = _collapse_spaces(el.get_text(" ", strip=True))
            if txt:
                current_text.append(txt)
    if current_text:
        sections.append({"heading": current_heading, "text": _collapse_spaces(" ".join(current_text))})
    return sections if sections else [{"heading": "", "text": _collapse_spaces(soup.get_text(" ", strip=True))}]

# ---------------- Fetch sitemap helper ----------------
def _fetch_sitemap_urls(base_url: str, timeout: int = 6) -> List[str]:
    candidates = [
        urljoin(base_url, "/sitemap.xml"),
        urljoin(base_url, "/sitemap_index.xml"),
        urljoin(base_url, "/sitemap/sitemap.xml"),
    ]
    urls = []
    headers = {"User-Agent": "general-crawler/1.0"}
    for s in candidates:
        try:
            r = requests.get(s, headers=headers, timeout=timeout)
            if r.status_code != 200:
                continue
            soup = BeautifulSoup(r.text, "xml")
            locs = [el.get_text().strip() for el in soup.find_all("loc")]
            if locs:
                for l in locs:
                    if l:
                        urls.append(l.strip())
                return list(dict.fromkeys(urls))
        except Exception:
            continue
    return []

# ---------------- Playwright wrapper ----------------
class PlaywrightEngine:
    def __init__(self, headless: bool = True, user_agent: str = DEFAULT_USER_AGENT):
        self.headless = headless
        self.user_agent = user_agent
        self._pw = None
        self._browser = None
        self._context = None
        self._page = None

    def start(self):
        if self._pw:
            return
        self._pw = sync_playwright().start()
        self._browser = self._pw.chromium.launch(headless=self.headless)
        self._context = self._browser.new_context(user_agent=self.user_agent)
        self._page = self._context.new_page()

    def stop(self):
        try:
            if self._context:
                self._context.close()
            if self._browser:
                self._browser.close()
            if self._pw:
                self._pw.stop()
        except Exception:
            pass
        finally:
            self._pw = None
            self._browser = None
            self._context = None
            self._page = None

# ---------------- Adapter ----------------
class Adapter:
    def __init__(self):
        self.max_pages = 20000
        self.max_pages_if_test = 50

    def scrape(self, engine, cfg: Dict[str, Any]):
        start_url = cfg.get("start_url") or cfg.get("url") or ""
        if not start_url:
            raise ValueError("start_url required in cfg")
        site_id = cfg.get("id") or _slugify_domain(start_url)
        site_name_cfg = cfg.get("name")
        test_mode = cfg.get("test_mode", False)
        pages_cap = self.max_pages_if_test if test_mode else (cfg.get("max_pages") or self.max_pages)

        if hasattr(engine, "start"):
            engine.start()
            page = engine._page
        else:
            pw = PlaywrightEngine(headless=False)
            pw.start()
            page = pw._page

        base_host = urlparse(start_url).netloc
        sitemap_urls = _fetch_sitemap_urls(start_url)
        queue = []
        if sitemap_urls:
            for u in sitemap_urls:
                if urlparse(u).netloc == base_host and not u.lower().endswith(".xml"):
                    queue.append(u)
        if not queue:
            queue = [start_url]

        seen = set()
        visited_count = 0

        try:
            while queue and visited_count < pages_cap:
                url = queue.pop(0)
                if not url or url in seen:
                    continue
                if urlparse(url).netloc != base_host:
                    seen.add(url)
                    continue
                try:
                    try:
                        page.goto(url, timeout=PAGE_GOTO_TIMEOUT_MS, wait_until="domcontentloaded")
                    except PWTimeoutError:
                        try:
                            page.goto(url, timeout=PAGE_GOTO_TIMEOUT_MS * 2)
                        except Exception:
                            seen.add(url)
                            continue
                    time.sleep(PAGE_WAIT_AFTER_LOAD)
                    html = page.content()
                    soup = BeautifulSoup(html, "html.parser")

                    for sel in ["header", "footer", "nav", ".site-header", ".site-footer", ".breadcrumbs", ".cookie-banner", ".cookie-notice", ".skip-link"]:
                        for el in soup.select(sel):
                            try:
                                el.decompose()
                            except Exception:
                                pass

                    # basic meta
                    meta = {}
                    title_el = soup.select_one("title")
                    meta["title"] = title_el.get_text(" ", strip=True) if title_el else ""
                    desc_el = soup.select_one("meta[name='description']") or soup.select_one("meta[property='og:description']")
                    meta["description"] = desc_el.get("content").strip() if desc_el and desc_el.get("content") else ""

                    # 1) Extract structured JSON-LD entities first — create one chunk per entity
                    page_id = _short_hash(url)
                    site_name = site_name_cfg or (meta.get("title")[:120] if meta.get("title") else _slugify_domain(start_url))
                    namespace = cfg.get("namespace") or "general"
                    metadata_base = {
                        "site_id": site_id,
                        "site_name": site_name,
                        "source_url": url,
                        "namespace": namespace,
                        "title": meta.get("title"),
                        "description": meta.get("description"),
                        "page_path": urlparse(url).path,
                    }

                    entities_emitted = []

                    jsonld = _extract_json_ld(soup)
                    for idx, ent in enumerate(jsonld):
                        try:
                            ent_type = ent.get("@type") or ent.get("type") or (ent.get("mainEntityOfPage", {}) and ent.get("mainEntityOfPage").get("@type")) or "Structured"
                            pieces = []
                            if isinstance(ent, dict):
                                for k in ("name", "headline", "description", "address", "telephone", "email", "url"):
                                    v = ent.get(k)
                                    if v:
                                        if isinstance(v, dict) and "name" in v:
                                            pieces.append(v.get("name"))
                                        else:
                                            pieces.append(str(v))
                                for k, v in ent.items():
                                    if k in ("@context", "@type"):
                                        continue
                                    if isinstance(v, (str, int, float)):
                                        if str(v).strip():
                                            pieces.append(f"{k}: {v}")
                            else:
                                pieces.append(str(ent))
                            text = _collapse_spaces(" • ".join([p for p in pieces if p]))
                            if not text:
                                text = _collapse_spaces(json.dumps(ent)[:1000])
                            ent_id = f"{site_id}_jsonld_{_short_hash(url + str(idx) + str(ent.get('name', '')))}"
                            meta_ent = dict(metadata_base)
                            meta_ent.update({"entity_type": "structured", "entity_subtype": ent_type, "entity_id": ent_id})
                            try:
                                meta_ent["jsonld"] = json.dumps(ent, ensure_ascii=False)[:_MAX_META_STR]
                            except Exception:
                                meta_ent["jsonld"] = str(ent)[:_MAX_META_STR]
                            chunk = {"id": ent_id, "text": text, "metadata": sanitize_metadata(meta_ent)}
                            yield chunk
                            entities_emitted.append(ent_id)
                        except Exception:
                            continue

                    for t in _extract_tables(soup, url, site_id):
                        if t["id"] not in entities_emitted:
                            yield t
                            entities_emitted.append(t["id"])
                    for d in _extract_definition_lists(soup, url, site_id):
                        if d["id"] not in entities_emitted:
                            yield d
                            entities_emitted.append(d["id"])
                    for c in _extract_list_cards(soup, url, site_id):
                        if c["id"] not in entities_emitted:
                            yield c
                            entities_emitted.append(c["id"])

                    text_page = _collapse_spaces(soup.get_text(" ", strip=True))
                    lower = text_page.lower()
                    heuristics_triggers = ["doctor", "specialist", "department", "team", "profile", "consultation", "clinic", "hospital", "service", "contact"]
                    if any(w in lower for w in heuristics_triggers):
                        for c in _extract_list_cards(soup, url, site_id):
                            if c["id"] not in entities_emitted:
                                yield c
                                entities_emitted.append(c["id"])

                    main_text = ""
                    if _HAS_READABILITY:
                        try:
                            main_text = _extract_main_using_readability(html) or ""
                        except Exception:
                            main_text = ""
                    if not main_text:
                        selectors = ["main", "article", ".article-body", ".content--main", ".page-content", ".post-content", ".entry-content", ".content"]
                        for sel in selectors:
                            el = soup.select_one(sel)
                            if el:
                                txt = _collapse_spaces(el.get_text(" ", strip=True))
                                if len(txt) > 120:
                                    main_text = txt
                                    break
                        if not main_text:
                            paragraphs = [p for p in (p.get_text(" ", strip=True) for p in soup.select("p")) if p and len(p) > 20]
                            if paragraphs:
                                main_text = _collapse_spaces(" ".join(paragraphs[:40]))
                            else:
                                main_text = _collapse_spaces(soup.get_text(" ", strip=True))

                    if len(main_text) > MAX_CHARS_PER_PAGE:
                        main_text = main_text[:MAX_CHARS_PER_PAGE]

                    sections = _extract_sections_by_headings(soup)
                    if not sections:
                        sections = [{"heading": "", "text": main_text}]
                    chunk_idx = 0
                    for sec in sections:
                        if not sec.get("text"):
                            continue
                        section_text = (sec.get("heading") + "\n" + sec.get("text")).strip() if sec.get("heading") else sec.get("text")
                        chunks = chunk_by_paragraphs(section_text, site_id, page_id, target=CHUNK_TARGET)
                        for c in chunks:
                            chunk_meta = dict(metadata_base)
                            chunk_meta.update({
                                "page_chunk_index": chunk_idx,
                                "page_id": page_id,
                                "entity_type": "page_section",
                                "entity_id": f"{site_id}_page_{page_id}",
                                "section_heading": sec.get("heading") or "",
                                "text_excerpt": (c["text"][:256] + "...") if len(c["text"]) > 256 else c["text"]
                            })
                            cid = c["id"]
                            yield {"id": cid, "text": c["text"], "metadata": sanitize_metadata(chunk_meta)}
                            chunk_idx += 1
                            if chunk_idx >= MAX_CHUNKS_PER_PAGE:
                                break
                        if chunk_idx >= MAX_CHUNKS_PER_PAGE:
                            break

                    seen.add(url)
                    visited_count += 1

                    found_links = []
                    for a in soup.select("a[href]"):
                        try:
                            href = a.get("href")
                            if not href:
                                continue
                            href = href.strip()
                            if not href:
                                continue
                            abs_url = href if href.startswith("http") else urljoin(url, href)
                            if abs_url.startswith("mailto:") or abs_url.startswith("tel:") or abs_url.startswith("javascript:"):
                                continue
                            if urlparse(abs_url).netloc != base_host:
                                continue
                            # skip assets
                            if re.search(r'\.(jpg|jpeg|png|gif|pdf|docx|zip|rar|svg|css|woff|ttf|json)(\?|$)', abs_url, re.I):
                                continue
                            found_links.append(abs_url.split("#")[0])
                        except Exception:
                            continue

                    uniq_links = []
                    for l in found_links:
                        if l not in seen and l not in queue and l not in uniq_links:
                            uniq_links.append(l)
                    sections_map = {}
                    for l in uniq_links:
                        path = urlparse(l).path
                        parts = [p for p in path.split("/") if p]
                        sec = ("/" + parts[0] + "/") if parts else "/"
                        sections_map.setdefault(sec, []).append(l)
                    enqueued = 0
                    for sec, links in sections_map.items():
                        to_add = links[:PER_SECTION_LIMIT]
                        for l in to_add:
                            if enqueued >= MAX_ENQUEUE_PER_PAGE:
                                break
                            if l not in seen and l not in queue:
                                queue.append(l)
                                enqueued += 1
                        if enqueued >= MAX_ENQUEUE_PER_PAGE:
                            break

                except KeyboardInterrupt:
                    print("general_adapter: KeyboardInterrupt - stopping crawl loop.")
                    break
                except Exception as e:
                    print("general_adapter: error processing", url, ":", e)
                    seen.add(url)
                    visited_count += 1
                    continue
        finally:
            try:
                if hasattr(engine, "stop"):
                    engine.stop()
            except Exception:
                pass
        return

# ---------------- Readability helper ----------------
def _extract_main_using_readability(html: str) -> str:
    if not _HAS_READABILITY:
        return ""
    try:
        doc = ReadabilityDocument(html)
        content = doc.summary()
        soup = BeautifulSoup(content, "html.parser")
        text = soup.get_text(" ", strip=True)
        return re.sub(r'\s+', ' ', text).strip()
    except Exception:
        return ""
