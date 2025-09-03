from flask import Blueprint, request, jsonify
from ..core.config import settings
from ..services.crawler import Crawler
from ..services.vectorstore import VectorStore

bp = Blueprint("admin", __name__)
_store: VectorStore | None = None

def store():
    global _store
    if _store is None:
        _store = VectorStore(settings)
    return _store

@bp.post("/crawl")
def admin_crawl():
    data = request.get_json(force=True)
    url = data["url"]
    c = Crawler(settings.CRAWL_OUTPUT_DIR)
    combined = c.crawl(url, max_pages=int(data.get("max_pages", 50)))
    site_meta = {"source_url": url}
    count = store().index_corpus(str(combined), site_meta)
    return jsonify({"ok": True, "chunks_indexed": count})
