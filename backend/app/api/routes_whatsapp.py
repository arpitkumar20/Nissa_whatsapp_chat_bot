from flask import Blueprint, request, jsonify, current_app
from ..services.vectorstore import VectorStore
from ..services.crawler import Crawler
from ..services.rag import generate_answer
from ..core.config import settings
from ..models.booking import BookingRequest
from ..services.whatsapp import send_whatsapp_text

bp = Blueprint("whatsapp", __name__)
_vstore: VectorStore | None = None

def vstore() -> VectorStore:
    global _vstore
    if _vstore is None:
        _vstore = VectorStore(settings)
    return _vstore

@bp.route("/", methods=["POST"])
def webhook():
    """
    Handles inbound WhatsApp messages.
    Commands:
      - crawl <url>
      - ask <question>
      - book <hotel|hospital> <name> <date>
      - tour <property|department>
      - help
    """
    from_number = request.form.get("From") or request.json.get("from")
    body = (request.form.get("Body") or request.json.get("body") or "").strip()
    current_app.logger.info("WA message from %s: %s", from_number, body)

    if body.lower().startswith("crawl "):
        url = body.split(" ", 1)[1].strip()
        c = Crawler(settings.CRAWL_OUTPUT_DIR)
        combined = c.crawl(url, max_pages=40)
        site_meta = {"source_url": url}
        count = vstore().index_corpus(str(combined), site_meta)
        send_whatsapp_text(from_number, f"✅ Crawled and indexed {count} chunks from {url}")
        return jsonify(ok=True)

    if body.lower().startswith("ask "):
        q = body.split(" ", 1)[1].strip()
        results = vstore().knn(q, settings.TOP_K)
        context = [r[1].text for r in results]
        ans = generate_answer(context, q)
        send_whatsapp_text(from_number, ans)
        return jsonify(ok=True)

    if body.lower().startswith("tour"):
        # in real impl, look up property-specific tour url from DB
        tour_url = settings.DEFAULT_TOUR_URL
        send_whatsapp_text(from_number, f"🎥 360° tour link: {tour_url}")
        return jsonify(ok=True)

    if body.lower().startswith("book "):
        # very simple parser; replace with NLP intent extraction later
        send_whatsapp_text(from_number, "📅 Booking request received. Our team will confirm shortly.")
        return jsonify(ok=True)

    send_whatsapp_text(from_number, "🤖 Commands: crawl <url>, ask <question>, book <...>, tour")
    return jsonify(ok=True)
