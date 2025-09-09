import os
import json
from typing import Any, Optional, List
from pinecone import Pinecone

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "nisaa-knowledge")

if not PINECONE_API_KEY:
    raise RuntimeError("PINECONE_API_KEY environment variable is required")

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX)

_MAX_META_STR = 16000

def _is_primitive(v: Any) -> bool:
    return isinstance(v, (str, int, float, bool))

def _sanitize_value(v: Any) -> Optional[Any]:
    if v is None:
        return None
    if _is_primitive(v):
        return v
    if isinstance(v, list):
        if all(isinstance(x, str) for x in v):
            return v
        if all(_is_primitive(x) for x in v):
            return [str(x) for x in v]
        try:
            s = json.dumps(v, ensure_ascii=False)
            return s[:_MAX_META_STR]
        except Exception:
            return None
    try:
        s = json.dumps(v, ensure_ascii=False)
        return s[:_MAX_META_STR]
    except Exception:
        try:
            s2 = str(v)
            return s2[:_MAX_META_STR]
        except Exception:
            return None

def _sanitize_metadata(metadata: dict) -> dict:
    out = {}
    if not metadata:
        return out
    for k, v in metadata.items():
        safe = _sanitize_value(v)
        if safe is None:
            continue
        out[k] = safe
    return out

def vector_db_upsert(vector_id: str, embedding: List[float], text: str, metadata: dict, namespace: str = "default"):
    meta = dict(metadata or {})
    meta["text"] = text
    safe_meta = _sanitize_metadata(meta)

    index.upsert(
        vectors=[
            {
                "id": vector_id,
                "values": embedding,
                "metadata": safe_meta,
            }
        ],
        namespace=namespace,
    )

def vector_db_query(query_embedding: List[float], top_k: int = 5, namespace: str = "default", filter: Optional[dict] = None):
    return index.query(
        vector=query_embedding,
        top_k=top_k,
        namespace=namespace,
        filter=filter,
        include_metadata=True,
    )
