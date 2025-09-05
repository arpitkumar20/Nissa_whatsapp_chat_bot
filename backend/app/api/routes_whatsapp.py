# import os
# from urllib.parse import unquote
# import requests
# from flask import Flask, request, jsonify
# from datetime import datetime
# import psycopg2
# from psycopg2.extras import RealDictCursor
# import google.generativeai as genai
# # ----------------------------
# # CONFIG
# # ----------------------------


# BASE_URL = "https://app-server.wati.io"
# API_KEY = os.getenv('WATI_APY_KEY')
# GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
# GEMINI_MODEL = os.getenv("GEMINI_MODEL")
# EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")

# # Init Gemini
# genai.configure(api_key=GOOGLE_API_KEY)
# gemini_model = genai.GenerativeModel("gemini-1.5-pro")
# embedding_model = "models/embedding-001"

# # Init Flask
# app = Flask(__name__)

# # ----------------------------
# # PostgreSQL + PGVector Connection
# # ----------------------------
# def get_db_connection():
#     print("[DB] Connecting to PostgreSQL...")
#     conn = psycopg2.connect(
#         host="localhost",
#         database="chatdb",
#         user="postgres",
#         password="root"
#     )
#     print("[DB] Connection successful")
#     return conn

# def send_whatsapp_message(phone_number, message) -> dict:
#     """
#     Send a WhatsApp session message using the WATI API.
#     """
#     # URL encode the message
#     encoded_message = unquote(message)

#     # url = f"{BASE_URL}/sendSessionMessage/{phone_number}?messageText={encoded_message}"
#     url = f"{BASE_URL}/api/v1/sendSessionMessage/{phone_number}?messageText={encoded_message}"

#     headers = {
#         "accept": "*/*",
#         "Authorization": f"Bearer {API_KEY}"
#     }

#     try:
#         response = requests.post(url, headers=headers)
#         if response.status_code == 200:
#             return response.json()
#     except requests.exceptions.RequestException as e:
#         return {"error": str(e)}

# # ----------------------------
# # Helper: Log messages in DB
# # ----------------------------
# def log_message(conversation_id, msg_id, sender, text, timestamp):
#     print(f"[DB] Logging message | sender={sender}, text={text}")
#     conn = get_db_connection()
#     cursor = conn.cursor()
#     cursor.execute(
#         """
#         INSERT INTO whatsapp_logs (conversation_id, message_id, sender, message_text, created_at)
#         VALUES (%s, %s, %s, %s, %s)
#         """,
#         (conversation_id, msg_id, sender, text, datetime.fromtimestamp(int(timestamp)))
#     )
#     conn.commit()
#     cursor.close()
#     conn.close()
#     print("[DB] Message logged successfully")

# # ----------------------------
# # Embeddings
# # ----------------------------
# # def get_embedding(text: str):
# #     print(f"[Embedding] Generating embedding for text: {text}")
# #     result = genai.embed_content(
# #         model=embedding_model,
# #         content=text
# #     )
# #     embedding = result["embedding"]
# #     print("[Embedding] Embedding generated")
# #     return embedding



# def get_embedding(text: str):
#     """
#     Generate embeddings for a given text using Gemini.
#     """
#     print(f"[Embedding] Generating embedding for text: {text[:50]}...")
#     result = genai.embed_content(model=EMBEDDING_MODEL, content=text)
#     embedding = result["embedding"]
#     print("[Embedding] Embedding generated.")
#     return embedding


# # import numpy as np

# # def cosine_similarity(vec1, vec2):
# #     v1 = np.array(vec1)
# #     v2 = np.array(vec2)
# #     return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

# # def semantic_search(user_embedding, top_k=3):
# #     conn = get_db_connection()
# #     cursor = conn.cursor(cursor_factory=RealDictCursor)
# #     cursor.execute("SELECT id, chunk, embedding FROM knowledge_base;")
# #     rows = cursor.fetchall()
# #     cursor.close()
# #     conn.close()

# #     # Compute cosine similarity in Python
# #     for row in rows:
# #         row["similarity"] = cosine_similarity(user_embedding, row["embedding"])

# #     # Sort by similarity (descending)
# #     rows = sorted(rows, key=lambda r: r["similarity"], reverse=True)
# #     return rows[:top_k]



# import os
# from pinecone import Pinecone

# # Initialize Pinecone client
# pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
# INDEX_NAME = os.getenv("PINECONE_INDEX", "nisaa-knowledge")
# index = pc.Index(INDEX_NAME)

# def vector_db_upsert(vector_id: str, embedding: list, metadata: dict, namespace: str = "default"):
#     """
#     Insert or update a vector in Pinecone with metadata and namespace segregation.
#     """
#     index.upsert(
#         vectors=[{"id": vector_id, "values": embedding, "metadata": metadata}],
#         namespace=namespace,
#     )
#     print(f"[Pinecone] Upserted vector ID: {vector_id}")

# def vector_db_query(query_embedding: list, top_k: int = 5, namespace: str = "default", filter: dict = None):
#     """
#     Query Pinecone for top_k similar vectors.
#     """
#     response = index.query(
#         vector=query_embedding,
#         top_k=top_k,
#         namespace=namespace,
#         filter=filter,
#         include_metadata=True,
#     )
#     return response

# # from embeddings import get_embedding
# # from vectordb import vector_db_query

# def semantic_search(query: str, top_k: int = 3, namespace: str = "default"):
#     """
#     Perform semantic search: embed query, fetch top-k results from Pinecone.
#     """
#     print(f"[Search] Running semantic search for: '{query}'")
    
#     # Step 1: Embed query
#     query_embedding = get_embedding(query)
    
#     # Step 2: Query Pinecone
#     results = vector_db_query(query_embedding, top_k=top_k, namespace=namespace)
    
#     # Step 3: Parse results
#     matches = []
#     for match in results["matches"]:
#         matches.append({
#             "id": match["id"],
#             "score": match["score"],
#             "metadata": match.get("metadata", {}),
#         })
    
#     print(f"[Search] Found {len(matches)} results.")
#     return matches




# # from embeddings import get_embedding
# # from vectordb import vector_db_upsert
# # from search import semantic_search

# # Example: Insert knowledge chunks
# docs = [
#     {"id": "1", "text": "Flask is a Python web framework.", "topic": "Python"},
#     {"id": "2", "text": "FastAPI is a modern web framework for APIs.", "topic": "Python"},
#     {"id": "3", "text": "Pinecone is a vector database for similarity search.", "topic": "Databases"}
# ]

# for doc in docs:
#     embedding = get_embedding(doc["text"])
#     vector_db_upsert(
#         vector_id=doc["id"],
#         embedding=embedding,
#         metadata={"text": doc["text"], "topic": doc["topic"]}
#     )

# # Example: Run search
# results = semantic_search("Tell me about Python frameworks", top_k=2)

# print("\n=== Search Results ===")
# for r in results:
#     print(f"ID: {r['id']} | Score: {r['score']:.4f} | Text: {r['metadata'].get('text')}")





# # ----------------------------
# # AI Response (Gemini + Context)
# # ----------------------------
# def get_ai_response(user_message):
#     print(f"[AI] User message: {user_message}")

#     # 1. Embed user query
#     user_embedding = get_embedding(user_message)

#     # 2. Search KB
#     top_chunks = semantic_search(user_embedding, top_k=3)
#     context_text = "\n".join([f"(ID {row['id']}) {row['chunk']}" for row in top_chunks])
#     retrieved_ids = [row["id"] for row in top_chunks]
#     print(f"[AI] Retrieved context IDs: {retrieved_ids}")

#     # 3. Build prompt
#     prompt = f"""
#     You are a helpful AI assistant.
#     User asked: {user_message}

#     Relevant knowledge base items:
#     {context_text}

#     Use the context to answer. If not enough info, say so.
#     """
#     print("[AI] Sending prompt to Gemini...")
#     response = gemini_model.generate_content(prompt)
#     print("[AI] Gemini response received")
#     return response.text, retrieved_ids

# # ----------------------------
# # Flask route - Webhook (WATI)
# # ----------------------------
# @app.route("/wati-webhook", methods=["POST"])
# def wati_webhook():
#     data = request.json
#     print("[Webhook] Incoming WATI Data:", data)

#     messages = data.get("messages", {}).get("items", [])
#     responses = []

#     for msg in messages:
#         text = msg.get("text")
#         msg_type = msg.get("type")
#         conversation_id = msg.get("conversationId")
#         msg_id = msg.get("id")
#         timestamp = msg.get("timestamp")
#         owner = msg.get("owner")  # False = user, True = bot

#         if msg_type == "text" and not owner and text:
#             print(f"[Webhook] Processing user message: {text}")

#             # 1. Log user message
#             log_message(conversation_id, msg_id, "user", text, timestamp)

#             # 2. AI response (RAG flow)
#             ai_reply, retrieved_ids = get_ai_response(text)

#             # 3. Log bot reply
#             log_message(conversation_id, f"bot-{msg_id}", "bot", ai_reply, timestamp)

#             # 4. Send reply to WhatsApp
#             phone_number = "919669092627"  # Replace with actual phone number mapping
#             send_resp = send_whatsapp_message(phone_number, ai_reply)

#             responses.append({
#                 "user_message": text,
#                 "ai_reply": ai_reply,
#                 "retrieved_ids": retrieved_ids,
#                 "wati_response": send_resp
#             })

#     print("[Webhook] Responses prepared:", responses)
#     return jsonify(responses)

# # ----------------------------
# # Run Flask
# # ----------------------------
# if __name__ == "__main__":
#     print("[Server] Starting Flask server on port 5000...")
#     app.run(port=5000, debug=True)










# '''
# project/
# │── embeddings.py     # Generate embeddings
# │── vectordb.py       # Pinecone upsert & query
# │── search.py         # Semantic search logic
# │── main.py           # Example usage


# '''
























import os
import logging
from datetime import datetime
from urllib.parse import quote_plus
import requests
from flask import Flask, request, jsonify

# DB
import psycopg2
from psycopg2.extras import RealDictCursor

# GenAI (Gemini)
import google.generativeai as genai

# Pinecone
import pinecone

# ----------------------------
# Logging
# ----------------------------
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("whatsapp-rag")

from dotenv import load_dotenv

load_dotenv()

# ----------------------------
# Config (environment variables)
# ----------------------------
WATI_API_KEY = os.getenv("WATI_API_KEY") or os.getenv("WATI_APY_KEY")  # support your previous typo as fallback
BASE_URL = "https://app-server.wati.io"
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "models/text-bison-001")  # change as needed
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/emb1")     # change to your embedding model name


# gemini_model = genai.GenerativeModel("gemini-1.5-pro")
# embedding_model = "models/embedding-001"

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")  # e.g. "us-west1-gcp"
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "nisaa-knowledge")

POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_DB = os.getenv("POSTGRES_DB", "chatdb")
POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
POSTGRES_PASS = os.getenv("POSTGRES_PASSWORD", "root")
POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", 5432))

# ----------------------------
# Initialize GenAI (Gemini)
# ----------------------------
if not GOOGLE_API_KEY:
    log.warning("GOOGLE_API_KEY not set — Gemeni API calls will fail until you set this env var.")
else:
    genai.configure(api_key=GOOGLE_API_KEY)
    # Try to instantiate a model handle if available
    try:
        # gemini_model = genai.GenerativeModel(GEMINI_MODEL)
        gemini_model = genai.GenerativeModel("gemini-1.5-pro")
    except Exception:
        # keep None and fallback to genai.generate below when necessary
        gemini_model = None

# ----------------------------
# Initialize Pinecone
# ----------------------------
# if not PINECONE_API_KEY or not PINECONE_ENV:
#     log.warning("Pinecone API key or env missing. Set PINECONE_API_KEY and PINECONE_ENV to use vector DB.")
#     pinecone_index = None
# else:
#     pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
#     pinecone_index = pinecone.Index(PINECONE_INDEX)



# ----------------------------
# Initialize Pinecone (v3 client)
# ----------------------------
from pinecone import Pinecone, ServerlessSpec

pc = None
pinecone_index = None

if not PINECONE_API_KEY:
    log.warning("Pinecone API key not set — skipping vector DB initialization.")
else:
    pc = Pinecone(api_key=PINECONE_API_KEY)

    # Create index if it doesn't exist
    existing_indexes = [idx.name for idx in pc.list_indexes()]
    if PINECONE_INDEX not in existing_indexes:
        log.info("Creating Pinecone index: %s", PINECONE_INDEX)
        pc.create_index(
            name=PINECONE_INDEX,
            dimension=768,   # 👈 adjust this to match your embedding size
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",   # or "gcp"
                region="us-west-2"
            )
        )

    # Connect to the index
    pinecone_index = pc.Index(PINECONE_INDEX)
    log.info("Connected to Pinecone index: %s", PINECONE_INDEX)



















# ----------------------------
# Flask app
# ----------------------------
app = Flask(__name__)

# ----------------------------
# Postgres DB helper
# ----------------------------
def get_db_connection():
    log.info("[DB] Connecting to PostgreSQL...")
    conn = psycopg2.connect(
        host=POSTGRES_HOST,
        database=POSTGRES_DB,
        user=POSTGRES_USER,
        password=POSTGRES_PASS,
        port=POSTGRES_PORT
    )
    log.info("[DB] Connection successful")
    return conn

def log_message(conversation_id, message_id, sender, text, ts):
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO whatsapp_logs (conversation_id, message_id, sender, message_text, created_at)
            VALUES (%s, %s, %s, %s, %s)
            """,
            (conversation_id, message_id, sender, text, datetime.fromtimestamp(int(ts)))
        )
        conn.commit()
        cur.close()
        conn.close()
        log.info("[DB] Message logged")
    except Exception as e:
        log.exception("Failed to log message to DB: %s", e)

# ----------------------------
# GenAI Embeddings wrapper
# ----------------------------
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
        raise ValueError(f"No embedding found in response: {resp}")
    except Exception as e:
        log.exception("Embedding generation failed: %s", e)
        raise

# ----------------------------
# Pinecone helper functions
# ----------------------------
def vector_db_upsert(vector_id: str, embedding: list, metadata: dict, namespace: str = "default"):
    if pinecone_index is None:
        log.error("Pinecone not initialized; cannot upsert.")
        return
    try:
        # Use dict form which is accepted by upsert
        pinecone_index.upsert(vectors=[{"id": vector_id, "values": embedding, "metadata": metadata}], namespace=namespace)
        log.info("[Pinecone] Upserted vector id=%s", vector_id)
    except Exception as e:
        log.exception("Pinecone upsert failed: %s", e)
        raise

def vector_db_query(query_embedding: list, top_k: int = 5, namespace: str = "default", filter: dict = None):
    if pinecone_index is None:
        log.error("Pinecone not initialized; cannot query.")
        return {"matches": []}
    try:
        resp = pinecone_index.query(vector=query_embedding, top_k=top_k, include_metadata=True, namespace=namespace, filter=filter)
        # resp may be an object or dict; normalize
        if isinstance(resp, dict):
            return resp
        # if resp has attribute 'matches'
        if hasattr(resp, "matches"):
            # convert to plain dict
            return {"matches": [m for m in resp.matches]}
        return {"matches": []}
    except Exception as e:
        log.exception("Pinecone query failed: %s", e)
        return {"matches": []}

# ----------------------------
# Semantic search convenience
# ----------------------------
def semantic_search_text(query: str, top_k: int = 3, namespace: str = "default"):
    """
    Embed the query and return the top_k matches (id, score, metadata).
    """
    try:
        q_emb = get_embedding(query)
    except Exception:
        log.error("Failed to embed query for semantic search.")
        return []

    resp = vector_db_query(q_emb, top_k=top_k, namespace=namespace)
    matches = []
    for m in resp.get("matches", []):
        # m may be dict-like or object-like
        if isinstance(m, dict):
            matches.append({
                "id": m.get("id"),
                "score": m.get("score"),
                "metadata": m.get("metadata", {})
            })
        else:
            # attempt to read attributes
            metadata = getattr(m, "metadata", {}) or {}
            matches.append({"id": getattr(m, "id", None), "score": getattr(m, "score", None), "metadata": metadata})
    return matches

# ----------------------------
# Gemini text generation wrapper
# ----------------------------
def generate_text_with_gemini(prompt: str, max_output_tokens: int = 512):
    # Try using gemini_model handle if available
    try:
        if gemini_model is not None and hasattr(gemini_model, "generate_content"):
            resp = gemini_model.generate_content(prompt)
            # resp may expose .text or be a dict
            if hasattr(resp, "text"):
                return resp.text
            if isinstance(resp, dict) and "text" in resp:
                return resp["text"]
            # fallback to string representation
            return str(resp)
        else:
            # Fallback: use genai.generate
            resp = genai.generate(model=GEMINI_MODEL, prompt=prompt, max_output_tokens=max_output_tokens)
            # resp often contains 'candidates' -> first -> 'content' or 'output'
            if isinstance(resp, dict):
                if "candidates" in resp and resp["candidates"]:
                    cand = resp["candidates"][0]
                    return cand.get("content") or cand.get("output") or str(cand)
                if "output" in resp:
                    return resp["output"]
            return str(resp)
    except Exception as e:
        log.exception("Gemini generation failed: %s", e)
        return "Sorry — I couldn't generate an answer right now."

# ----------------------------
# High-level RAG responder
# ----------------------------
def get_ai_response(user_message: str, top_k: int = 3):
    log.info("[AI] Generating response for user message (short preview): %s", user_message[:80])
    # 1) retrieve top context chunks from Pinecone
    matches = semantic_search_text(user_message, top_k=top_k)
    print(">>>>>>>>>>>>>>>>>>>>matches>>>>>>>>>>>>>>>>>>>>",matches)
    # Build context text
    context_lines = []
    for m in matches:
        meta = m.get("metadata", {})
        # common metadata key we store: 'text' or 'chunk'
        chunk_text = meta.get("text") or meta.get("chunk") or meta.get("content") or ""
        source = meta.get("source") or meta.get("topic") or ""
        context_lines.append(f"[{m.get('id')}] {source}: {chunk_text}")
    context_text = "\n".join(context_lines) or "(no relevant KB items found)"
    print(">>>>>>>>>>>>>>>>>>>>context_text>>>>>>>>>>>>>>>>>>>>",context_text)

    # 2) Build prompt
    prompt = (
        "You are a helpful assistant. Use the provided context from the knowledge base to answer the user's question.\n\n"
        f"Context:\n{context_text}\n\n"
        f"User question: {user_message}\n\n"
        "If the context doesn't contain enough information, say you don't know and be concise."
    )

    # 3) Generate via Gemini
    ai_text = generate_text_with_gemini(prompt)
    retrieved_ids = [m.get("id") for m in matches]
    return ai_text, retrieved_ids

# ----------------------------
# WATI sending helper
# ----------------------------
def send_whatsapp_message(phone_number: str, message: str) -> dict:
    """
    Sends a session message to WATI. Try the query-string style first, fall back to JSON.
    """
    if WATI_API_KEY is None:
        return {"error": "WATI API key not configured"}

    encoded_message = quote_plus(message)
    url_qs = f"{BASE_URL}/api/v1/sendSessionMessage/{phone_number}?messageText={encoded_message}"
    headers = {"accept": "*/*", "Authorization": f"Bearer {WATI_API_KEY}"}

    try:
        r = requests.post(url_qs, headers=headers, timeout=10)
        if r.status_code == 200:
            return r.json()
        # Fallback: try JSON body format (some WATI endpoints prefer JSON)
        json_payload = {"phone": phone_number, "message": message}
        r2 = requests.post(f"{BASE_URL}/api/v1/sendMessage", headers={**headers, "Content-Type": "application/json"}, json=json_payload, timeout=10)
        return {"status_code": r.status_code, "qs_response": r.text, "fallback_status": r2.status_code, "fallback_response": r2.text}
    except Exception as e:
        log.exception("Failed to send message to WATI: %s", e)
        return {"error": str(e)}

# ----------------------------
# Example: Upsert some docs into Pinecone (call at startup or manually)
# ----------------------------
# def bootstrap_example_docs():
#     docs = [
#         {"id": "1", "text": "Flask is a Python web framework.", "topic": "Python"},
#         {"id": "2", "text": "FastAPI is a modern Python web framework optimized for APIs.", "topic": "Python"},
#         {"id": "3", "text": "Pinecone is a vector database for similarity search.", "topic": "Databases"}
#     ]
#     if pinecone_index is None:
#         log.info("Skipping bootstrap: Pinecone not initialized.")
#         return

#     for doc in docs:
#         try:
#             emb = get_embedding(doc["text"])
#             vector_db_upsert(vector_id=doc["id"], embedding=emb, metadata={"text": doc["text"], "topic": doc["topic"]})
#         except Exception:
#             log.exception("Failed to upsert doc id=%s", doc["id"])

# # Call bootstrap (optional)
# try:
#     bootstrap_example_docs()
# except Exception:
#     log.exception("Bootstrap failed.")

# ----------------------------
# Flask route - WATI webhook
# ----------------------------
@app.route("/wati-webhook", methods=["POST"])
def wati_webhook():
    data = request.get_json(force=True, silent=True) or {}
    log.info("[Webhook] Incoming data keys: %s", list(data.keys()))
    # WATI can deliver different shapes; typical shape: {"messages": {"items": [ ... ]}}
    messages = []
    if isinstance(data.get("messages"), dict) and isinstance(data["messages"].get("items"), list):
        messages = data["messages"]["items"]
    elif isinstance(data.get("messages"), list):
        messages = data["messages"]
    else:
        # fallback - maybe the webhook directly sent a single message payload
        messages = [data]

    responses = []
    for msg in messages:
        text = msg.get("text") or msg.get("body") or (msg.get("message") and msg["message"].get("text"))
        msg_type = msg.get("type")
        conversation_id = msg.get("conversationId") or msg.get("conversation_id") or msg.get("chatId")
        msg_id = msg.get("id") or msg.get("_id") or msg.get("messageId")
        timestamp = msg.get("timestamp") or int(datetime.utcnow().timestamp())
        owner = msg.get("owner")  # False = user, True = bot, depends on WATI shape
        from_number = msg.get("from") or msg.get("sender") or msg.get("phone") or None

        if msg_type == "text" or (text and not owner):
            # 1) Log
            log_message(conversation_id, msg_id, "user", text, timestamp)

            # 2) Get AI reply
            ai_reply, retrieved_ids = get_ai_response(text)

            # 3) Log bot reply
            bot_msg_id = f"bot-{msg_id}"
            log_message(conversation_id, bot_msg_id, "bot", ai_reply, timestamp)

            # 4) Send via WATI (if we can detect phone number)
            # If webhook didn't provide `from_number`, you should map conversation_id -> phone in your app
            phone_number_to_send = from_number or os.getenv("TEST_RECIPIENT_PHONE") or "919669092627"
            wati_resp = send_whatsapp_message(phone_number_to_send, ai_reply)

            responses.append({
                "user_message": text,
                "ai_reply": ai_reply,
                "retrieved_ids": retrieved_ids,
                "wati_response": wati_resp
            })

    return jsonify(responses)

# ----------------------------
# Simple health endpoint
# ----------------------------
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "time": datetime.utcnow().isoformat()})

# ----------------------------
# Run
# ----------------------------
if __name__ == "__main__":
    log.info("Starting Flask server on port 5000...")
    app.run(port=5000, debug=True)
