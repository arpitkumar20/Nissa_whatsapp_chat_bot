import importlib
from config.generate_site_config import generate_configs
from crawler.crawler_core import CrawlEngine
from scraper_service.embeddings.service_test import embed_text
from vectordb.client import vector_db_upsert



START_SITES = [
    "https://www.tajhotels.com/en-in",
    # Add more start sites
]

SITE_CONFIGS = generate_configs(START_SITES)

def run_scraper(site_key, cfg):
    adapter_mod = importlib.import_module(cfg["adapter"])
    Adapter = getattr(adapter_mod, "Adapter")
    adapter = Adapter()
    print(f"\n>>> Running scraper for {site_key} ({cfg['start_url']}) ...")
    engine = CrawlEngine(start_url=cfg["start_url"], site_id=cfg["id"], site_name=cfg["name"],
                         max_pages=cfg.get("max_pages", 50), test_mode=cfg.get("test_mode", True), headless=False)
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

# def main():
#     for key, cfg in SITE_CONFIGS.items():
#         try:
#             run_scraper(key, cfg)
#         except Exception as e:
#             print("Error running scraper for", key, e)

# if __name__ == "__main__":
#     main()
