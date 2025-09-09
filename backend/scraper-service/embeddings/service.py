import os
from google import genai
from google.genai.types import EmbedContentConfig
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
