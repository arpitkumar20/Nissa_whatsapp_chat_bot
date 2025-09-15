import json
from app.web_scraping.web_scraper import WebsiteProcessor

 
def web_scraping(url):
    # Website to scrape
    WEBSITE_URL = url

    print("="*60)
    print("RAG DATA PREPARATION PIPELINE")
    print("="*60)

    print("\n📦 Initializing processor...")
    processor = WebsiteProcessor(WEBSITE_URL)

    print("\n🌐 Scraping website...")
    documents = processor.scrape_website()
    print(f"   ✅ Scraped {len(documents)} pages")

    print("\n🧹 Cleaning documents...")
    cleaned_docs = processor.clean_documents()
    print(f"   ✅ Cleaned {len(cleaned_docs)} documents")

    print("\n📄 Creating chunks...")
    chunks = processor.create_chunks(chunk_size=1000, chunk_overlap=200)
    print(f"   ✅ Created {len(chunks)} chunks")

    print("\n" + "="*60)
    print("PROCESSING COMPLETE!")
    print("="*60)

    embedding_path = processor.website_folder / "chunks" / "embedding_ready.json"
    if embedding_path.exists():
        with open(embedding_path, 'r', encoding='utf-8') as f:
            embedding_data = json.load(f)
        print(f"\n📊 Statistics:")
        print(f"   • Total chunks ready for embedding: {embedding_data['total_chunks']}")
        print(f"   • Average chunk size: ~1000 characters")
        print(f"   • Ready for vector database: YES ✅")
    else:
        print("Embedding data not found.")