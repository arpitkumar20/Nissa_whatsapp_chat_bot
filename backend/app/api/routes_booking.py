import pinecone
from pinecone import (
    Pinecone,
    ServerlessSpec,
    CloudProvider,
    AwsRegion,
    VectorType
)

import numpy as np
import requests

PINECONE_API_KEY= "pcsk_5ZJEVn_K6FrjVje2XZnYuqxyhfJVYDVKuKg5A6RZc4UWaPKNzARdQxKK82o2xNc82paxBk"
# PINECONE_ENV="us-west1-gcp"
PINECONE_ENV = "us-east-1"
PINECONE_INDEX="nisaa-knowledge"
NAMESPACE="hospital"
GOOGLE_API_KEY="AIzaSyBT7ulr-_i-O1Z42rKLDr8ZJJjh9v52StM"
GEMINI_MODEL="gemini-1.5-pro"
# EMBEDDING_MODEL="models/embedding-001"
EMBEDDING_MODEL="models/llama-text-embed-v2"


# # Initialize Pinecone
# pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)

# # Check available indexes to ensure correct setup
# indexes = pc.list_indexes()
# print(f"Available Indexes: {indexes}")

# # Connect to your Pinecone index
# index = pc.Index(PINECONE_INDEX)

# # Step 1: Generate embedding from GenAI (Gemini API)
# query_text = "give me all doctors list"
# # genai_url = "https://api.gemini.google.com/v1/embed/text"

# # genai_url = f"https://generativelanguage.googleapis.com/v1beta2/{EMBEDDING_MODEL}:embedText"
# genai_url = f"https://generativelanguage.googleapis.com/v1beta2/{EMBEDDING_MODEL}:embedText"


# headers = {
#     "Content-Type": "application/json"
# }

# payload = {
#     "text": query_text
# }

# params = {
#     "key": GOOGLE_API_KEY
# }

# response = requests.post(genai_url, headers=headers, json=payload, params=params)
# response.raise_for_status()

# embedding_data = response.json()
# query_vector = embedding_data['embedding']  # Correct key path for embedding result

# # Step 2: Query Pinecone Vector DB
# query_response = index.query(
#     vector=query_vector,
#     top_k=5,
#     namespace=NAMESPACE,
#     include_metadata=True
# )

# # Print results
# print("Query Results:")
# for match in query_response['matches']:
#     print(f"ID: {match['id']}, Score: {match['score']}, Metadata: {match.get('metadata')}")




import google.generativeai as genai
from pinecone import Pinecone
import os




# ------------------------------
# Initialize GenAI SDK
# ------------------------------
genai.configure(api_key=GOOGLE_API_KEY)

def get_embedding(text: str):
    """
    Returns a list[float] embedding for `text` using the configured embedding model.
    """
    try:
        resp = genai.embed_content(model="models/embedding-001", content=text)
        # Different response shapes can occur; try common keys
        if isinstance(resp, dict):
            if "embedding" in resp:
                return resp["embedding"]
            if "embeddings" in resp and resp["embeddings"]:
                # sometimes returns list of embeddings
                return resp["embeddings"][0]
        # If resp is an object with attributes
        if hasattr(resp, "embedding"):
            return resp.embedding
        # fallback
        raise ValueError(f"No embedding found in response:+{resp}")
    except Exception as e:
        raise

# ------------------------------
# Step 1: Generate Embedding
# ------------------------------
query_text = "give me all doctors list"

# model = genai.GenerativeModel(EMBEDDING_MODEL)

# response = model.t .embed_text(query_text)
query_vector = get_embedding(query_text)  # embedding vector list
# ------------------------------
# Step 2: Initialize Pinecone
# ------------------------------
pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)

# Check available indexes
indexes = pc.list_indexes()
print(f"Available Indexes: {indexes}")

# Connect to Pinecone index
index = pc.Index(PINECONE_INDEX)

# ------------------------------
# Step 3: Query Pinecone
# ------------------------------
query_response = index.query(
    vector=query_vector,
    top_k=5,
    namespace=NAMESPACE,
    include_metadata=True
)

# ------------------------------
# Step 4: Print results
# ------------------------------
print("Query Results:")
for match in query_response['matches']:
    print(f"ID: {match['id']}, Score: {match['score']}, Metadata: {match.get('metadata')}")
