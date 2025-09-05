# from flask import Blueprint, request, jsonify
# from ..models.booking import BookingRequest, BookingConfirmation

# bp = Blueprint("booking", __name__)

# @bp.post("/create")
# def create_booking():
#     data = request.get_json(force=True)
#     br = BookingRequest.model_validate(data)
#     # mock integration; replace with HMS/hotel API call
#     confirmation = BookingConfirmation(
#         booking_id="BK-" + br.customer_phone[-4:],
#         status="CONFIRMED",
#         provider="mock",
#         details=br.model_dump()
#     )
#     return jsonify(confirmation.model_dump())


# import os
# from pinecone import Pinecone

# # ----------------------------
# # Load env variables
# # ----------------------------

# PINECONE_API_KEY= "pcsk_5ZJEVn_K6FrjVje2XZnYuqxyhfJVYDVKuKg5A6RZc4UWaPKNzARdQxKK82o2xNc82paxBk"
# PINECONE_ENV="us-west1-gcp"
# PINECONE_INDEX="nisaa-knowledge"
# # ----------------------------
# # Connect to Pinecone
# # ----------------------------
# pc = Pinecone(api_key=PINECONE_API_KEY)
# index = pc.Index(PINECONE_INDEX)

# # ----------------------------
# # Example 1: Query with embedding
# # ----------------------------
# # Suppose you have an embedding vector from Gemini (replace with real embedding)
# example_vector = [0.01] * 768  # adjust length to match your index dimension

# query_result = index.query(
#     vector=example_vector,
#     top_k=5,          # number of nearest matches
#     include_values=True,
#     include_metadata=True
# )

# print("🔎 Query Results:")
# for match in query_result["matches"]:
#     print(f"ID: {match['id']}")
#     print(f"Score: {match['score']}")
#     print(f"Metadata: {match.get('metadata')}")
#     print("------")

# # ----------------------------
# # Example 2: Fetch by IDs (already stored data)
# # ----------------------------
# fetch_result = index.fetch(ids=["doc1", "doc2"])  # replace with real IDs
# print("📦 Fetch Results:", fetch_result)






# from pinecone import Pinecone

# # Initialize Pinecone
# pc = Pinecone(api_key=PINECONE_API_KEY)

# # Connect to your index
# index_name = PINECONE_INDEX
# index = pc.Index(index_name)

# # List of IDs to fetch
# ids_to_fetch = ["id1", "id2", "id3"]

# # Fetch the vectors
# response = index.fetch(ids=ids_to_fetch)

# # Print fetched vectors
# print(response)














# from pinecone import Pinecone
# import numpy as np

# # Initialize Pinecone
# pc = Pinecone(api_key=PINECONE_API_KEY)

# # Connect to your index
# index_name = PINECONE_INDEX
# index = pc.Index(index_name)

# # Get index statistics
# stats = index.describe_index_stats()

# # Pinecone returns stats as a dict
# # Example: {'namespaces': {'': {'vector_count': 123}}, 'dimension': 1536}
# num_dimensions = stats["dimension"]
# total_vector_count = sum(ns["vector_count"] for ns in stats["namespaces"].values())

# # Create a dummy query vector (zeros)
# dummy_vector = np.zeros(num_dimensions).tolist()

# # Query with a high top_k (max 10000 allowed)
# query_results = index.query(
#     vector=dummy_vector,
#     top_k=5,
#     include_values=True,
#     include_metadata=True,
# )

# for match in query_results["matches"]:
#     print(">>>>>>>>>>>>>>",match)


# # Extract IDs
# retrieved_ids = [match["id"] for match in query_results["matches"]]

# # Fetch vectors (in batches if needed)
# if retrieved_ids:
#     fetched_vectors = index.fetch(ids=retrieved_ids)
#     print(fetched_vectors)
