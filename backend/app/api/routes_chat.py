from flask import Blueprint, request, jsonify
from ..core.config import settings
from ..services.vectorstore import VectorStore
from ..services.rag import generate_answer

bp = Blueprint("chat", __name__)
_store: VectorStore | None = None

def store():
    global _store
    if _store is None:
        _store = VectorStore(settings)
    return _store

@bp.post("/query")
def query():
    data = request.get_json(force=True)
    question = data.get("question","")
    top_k = int(data.get("top_k", settings.TOP_K))
    results = store().knn(question, top_k)
    context = [r[1].text for r in results]
    answer = generate_answer(context, question)
    return jsonify({"answer": answer, "hits": [
        {"score": s, "meta": r.metadata} for s, r in results
    ]})
