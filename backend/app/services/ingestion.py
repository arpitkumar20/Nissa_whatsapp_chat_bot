from pathlib import Path
from typing import Iterable
import re, math

def split_text_to_chunks(text: str, max_tokens: int = 500) -> list[str]:
    # simple sentence-based splitter with soft limit
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks, cur = [], ""
    for s in sentences:
        if len((cur + " " + s).split()) > max_tokens:
            if cur.strip():
                chunks.append(cur.strip())
            cur = s
        else:
            cur = (cur + " " + s).strip()
    if cur.strip():
        chunks.append(cur.strip())
    return chunks

def load_chunks_from_file(path: Path, max_tokens: int = 500) -> list[str]:
    text = path.read_text(encoding="utf-8")
    return split_text_to_chunks(text, max_tokens=max_tokens)
