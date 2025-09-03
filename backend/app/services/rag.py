# from pydantic import BaseModel
# from .core.config import settings

# class SentenceEmbedder:
#     def __init__(self, model_name: str):
#         from sentence_transformers import SentenceTransformer
#         self.model = SentenceTransformer(model_name)

#     @property
#     def dim(self) -> int:
#         return self.model.get_sentence_embedding_dimension()

#     def encode(self, texts: list[str], normalize_embeddings: bool = True):
#         import numpy as np
#         embs = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=normalize_embeddings)
#         return embs

# def get_embedder(cfg=settings):
#     if cfg.EMBEDDINGS_PROVIDER == "sentence":
#         emb = SentenceEmbedder(cfg.SENTENCE_MODEL)
#         return emb, emb.dim
#     raise NotImplementedError("Only sentence-transformers enabled for now.")

# # generation
# # def generate_answer(context: list[str], question: str) -> str:
# #     """
# #     Keep it simple: template + optional OpenAI.
# #     """
# #     import os
# #     key = os.getenv("OPENAI_API_KEY")
# #     if key:
# #         from openai import OpenAI
# #         client = OpenAI()
# #         system = "You are NISAA, a helpful assistant for hospitals/hotels. Answer strictly from context."
# #         content = [
# #             {"role": "system", "content": system},
# #             {"role": "user", "content": f"Question: {question}\n\nContext:\n" + "\n\n---\n\n".join(context)}
# #         ]
# #         resp = client.chat.completions.create(model="gpt-4o-mini", messages=content, temperature=0.2)
# #         return resp.choices[0].message.content.strip()
# #     # fallback: extractive-ish answer
# #     import textwrap
# #     top = "\n\n".join(context[:2])[:1200]
# #     return textwrap.shorten(f"(best-effort from context) {top}", width=800, placeholder="…")
