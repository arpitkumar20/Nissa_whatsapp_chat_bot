import importlib
from scraper_service.config.generate_site_config import generate_configs
from scraper_service.crawler.crawler_core import CrawlEngine
# from scraper_service.embeddings.service_test import embed_text
from scraper_service.vectordb.client import vector_db_upsert



import os
from google import genai
from google.genai.types import EmbedContentConfig
import google.generativeai as genai


import os
from dotenv import load_dotenv  

load_dotenv()
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

def embed_text(text: str):
    response = client.models.embed_content(
        model=os.getenv("EMBEDDING_MODEL", "models/embedding-001"),
        contents=[text],
        config=EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")
    )
    return response.embeddings[0].values


def run_scraper(site_key, cfg):
    adapter_mod = importlib.import_module(cfg["adapter"])
    Adapter = getattr(adapter_mod, "Adapter")
    adapter = Adapter()
    print(f"\n>>> Running scraper for {site_key} ({cfg['start_url']}) ...")
    engine = CrawlEngine(
        start_url=cfg["start_url"],
        site_id=cfg["id"],
        site_name=cfg["name"],
        max_pages=cfg.get("max_pages", 50),
        test_mode=cfg.get("test_mode", True),
        headless=False
    )
    for chunk in adapter.scrape(engine, cfg):
        try:
            embedding = embed_text(chunk["text"])
            vector_db_upsert(
                vector_id=chunk["id"],
                embedding=embedding,
                text=chunk["text"],
                metadata=chunk.get("metadata", {}),
                namespace=cfg["namespace"]
            )
            print(f"Stored: {chunk['id']} → {chunk['text'][:120]}...")
        except Exception as e:
            print("Upsert/embed error:", e)


def scraper_fun(url):
    print(">>>>>>>>>>>>okkk>>>>>>>>>>>", url)
    start_sites = [url]
    site_configs = generate_configs(start_sites)

    for key, cfg in site_configs.items():
        try:
            run_scraper(key, cfg)
        except Exception as e:
            print("Error running scraper for", key, e)

    return {"status": "accepted", "url": url}

