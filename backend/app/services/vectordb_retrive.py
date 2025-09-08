import os
import logging
import google.generativeai as genai
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()

PINECONE_API_KEY= os.getenv('PINECONE_API_KEY')
PINECONE_ENV= os.getenv('PINECONE_ENV')
PINECONE_INDEX= os.getenv('PINECONE_INDEX')
NAMESPACE= os.getenv('NAMESPACE')

GOOGLE_API_KEY= os.getenv('GOOGLE_API_KEY')
GEMINI_MODEL= os.getenv('GEMINI_MODEL')
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL')

# # PINECONE_ENV="us-west1-gcp"
# PINECONE_ENV = "us-east-1"
# PINECONE_INDEX="nisaa-knowledge"
# NAMESPACE="hospital"
# GOOGLE_API_KEY="AIzaSyBT7ulr-_i-O1Z42rKLDr8ZJJjh9v52StM"
# GEMINI_MODEL="gemini-1.5-pro"
# # EMBEDDING_MODEL="models/embedding-001"
# EMBEDDING_MODEL="models/llama-text-embed-v2"


# ------------------------------
# Step 0: Configure logging
# ------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# ------------------------------
# Step 1: Configure GenAI
# ------------------------------
genai.configure(api_key=GOOGLE_API_KEY)

# ------------------------------
# Step 2: Embedding function
# ------------------------------
def generate_embedding(text: str, model: str = EMBEDDING_MODEL) -> list[float]:
    """
    Generate embedding vector for a given text using GenAI.
    """
    try:
        logger.info("Generating embedding for text: %s", text)
        resp = genai.embed_content(model=model, content=text)
        
        if isinstance(resp, dict):
            if "embedding" in resp:
                return resp["embedding"]
            if "embeddings" in resp and resp["embeddings"]:
                return resp["embeddings"][0]
        if hasattr(resp, "embedding"):
            return resp.embedding

        raise ValueError(f"No embedding found in response: {resp}")
    except Exception as e:
        logger.error("Error generating embedding: %s", e)
        raise

# ------------------------------
# Step 3: Query Pinecone function
# ------------------------------
def query_pinecone(query_text: str, top_k: int = 5, namespace: str = NAMESPACE) -> list[dict]:
    """
    Generate embedding for `query_text` and query Pinecone index.
    Returns a list of matching items with score and metadata.
    """
    try:
        logger.info("Starting Pinecone query for text: %s", query_text)
        
        # Generate embedding
        query_vector = generate_embedding(query_text)
        logger.info("Embedding generated successfully. Vector length: %d", len(query_vector))

        # Initialize Pinecone
        pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
        index = pc.Index(PINECONE_INDEX)
        logger.info("Connected to Pinecone index: %s", PINECONE_INDEX)

        # Query Pinecone
        query_response = index.query(
            vector=query_vector,
            top_k=top_k,
            namespace=namespace,
            include_metadata=True
        )

        results = []
        for match in query_response.get('matches', []):
            result_dict = {
                "id": match['id'],
                "score": match['score'],
                **match.get('metadata', {})
            }
            results.append(result_dict)

        logger.info("Query returned %d results", len(results))
        return results

    except Exception as e:
        logger.error("Error querying Pinecone: %s", e)
        raise

# ------------------------------
# Step 4: Run query
# ------------------------------
# if __name__ == "__main__":
#     query_text = "give me only one cancer specialist doctor name"
#     results = query_pinecone(query_text)
    
#     print("Query Results:")
#     for r in results:
#         print(r)