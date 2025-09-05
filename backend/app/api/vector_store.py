import os
import requests
import psycopg2
from flask import Flask, request, jsonify
from urllib.parse import quote_plus
from datetime import datetime
import google.generativeai as genai
from pinecone import Pinecone, ServerlessSpec



from dotenv import load_dotenv

load_dotenv()

# ----------------------------
# Config
# ----------------------------
# GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
# PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
# PINECONE_INDEX = os.getenv("PINECONE_INDEX", "whatsapp-index")

# POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
# POSTGRES_DB = os.getenv("POSTGRES_DB", "chatdb")
# POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
# POSTGRES_PASS = os.getenv("POSTGRES_PASS", "root")
# POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")

# WATI_API_KEY = os.getenv("WATI_API_KEY")
# BASE_URL = os.getenv("WATI_BASE_URL", "https://app-server.wati.io")






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



# # ----------------------------
# # Init Gemini + Embeddings
# # ----------------------------
# genai.configure(api_key=GOOGLE_API_KEY)
# gemini_model = genai.GenerativeModel("gemini-1.5-pro")

# def get_embedding(text: str):
#     """Generate embedding for text"""
#     resp = genai.embed_content(model="models/embedding-001", content=text)
#     return resp["embedding"]

# # ----------------------------
# # Init Pinecone
# # ----------------------------
# pinecone_index = None
# if PINECONE_API_KEY:
#     pc = Pinecone(api_key=PINECONE_API_KEY)
#     existing = [idx.name for idx in pc.list_indexes()]
#     if PINECONE_INDEX not in existing:
#         pc.create_index(
#             name=PINECONE_INDEX,
#             dimension=768,
#             metric="cosine",
#             spec=ServerlessSpec(cloud="aws", region="us-west-2")
#         )
#     pinecone_index = pc.Index(PINECONE_INDEX)

# def vector_db_upsert(vector_id, embedding, metadata):
#     if pinecone_index:
#         pinecone_index.upsert(
#             vectors=[{"id": vector_id, "values": embedding, "metadata": metadata}],
#             namespace="default"
#         )

# def semantic_search(query: str, top_k: int = 3):
#     if not pinecone_index:
#         return []
#     q_emb = get_embedding(query)
#     resp = pinecone_index.query(vector=q_emb, top_k=top_k, include_metadata=True)
#     return resp.get("matches", [])

# # ----------------------------
# # Postgres
# # ----------------------------
# def get_db_connection():
#     return psycopg2.connect(
#         host=POSTGRES_HOST,
#         database=POSTGRES_DB,
#         user=POSTGRES_USER,
#         password=POSTGRES_PASS,
#         port=POSTGRES_PORT
#     )

# def log_message(conversation_id, message_id, sender, text, ts):
#     conn = get_db_connection()
#     cur = conn.cursor()
#     cur.execute(
#         """INSERT INTO whatsapp_logs 
#            (conversation_id, message_id, sender, message_text, created_at)
#            VALUES (%s, %s, %s, %s, %s)""",
#         (conversation_id, message_id, sender, text, datetime.fromtimestamp(int(ts)))
#     )
#     conn.commit()
#     cur.close()
#     conn.close()

# # ----------------------------
# # RAG Response
# # ----------------------------
# def get_ai_response(user_message: str, msg_id: str):
#     # Upsert user query into Pinecone
#     emb = get_embedding(user_message)
#     vector_db_upsert(msg_id, emb, {"text": user_message})

#     # Retrieve context
#     matches = semantic_search(user_message)
#     context = "\n".join([m["metadata"].get("text", "") for m in matches]) or "(no context found)"

#     # Generate reply
#     prompt = f"Context:\n{context}\n\nUser: {user_message}\n\nAnswer concisely."
#     resp = gemini_model.generate_content(prompt)
#     return resp.text if hasattr(resp, "text") else str(resp)

# # ----------------------------
# # WATI Helper
# # ----------------------------
# def send_whatsapp_message(phone_number: str, message: str):
#     encoded = quote_plus(message)
#     url = f"{BASE_URL}/api/v1/sendSessionMessage/{phone_number}?messageText={encoded}"
#     headers = {"Authorization": f"Bearer {WATI_API_KEY}"}
#     r = requests.post(url, headers=headers, timeout=10)
#     return r.json() if r.status_code == 200 else {"error": r.text}

# # ----------------------------
# # Flask App
# # ----------------------------
# app = Flask(__name__)

# @app.route("/wati-webhook", methods=["POST"])
# def wati_webhook():
#     data = request.get_json(force=True, silent=True) or {}
#     messages = data.get("messages", {}).get("items", [data])

#     responses = []
#     for msg in messages:
#         text = msg.get("text") or msg.get("body")
#         msg_id = msg.get("id") or str(datetime.utcnow().timestamp())
#         conversation_id = msg.get("conversationId", "conv-default")
#         timestamp = msg.get("timestamp") or int(datetime.utcnow().timestamp())
#         from_number = msg.get("from") or os.getenv("TEST_RECIPIENT_PHONE")

#         if text:
#             # Log user message
#             log_message(conversation_id, msg_id, "user", text, timestamp)

#             # Get AI response
#             ai_reply = get_ai_response(text, msg_id)

#             # Log bot reply
#             log_message(conversation_id, f"bot-{msg_id}", "bot", ai_reply, timestamp)

#             # Send reply
#             wati_resp = send_whatsapp_message(from_number, ai_reply)

#             responses.append({"user": text, "bot": ai_reply, "wati": wati_resp})

#     return jsonify(responses)

# @app.route("/health", methods=["GET"])
# def health():
#     return jsonify({"status": "ok", "time": datetime.utcnow().isoformat()})

# if __name__ == "__main__":
#     app.run(port=5000, debug=True)









import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
log = logging.getLogger(__name__)

# ----------------------------
genai.configure(api_key=GOOGLE_API_KEY)
gemini_model = genai.GenerativeModel("gemini-1.5-pro")

def get_embedding(text: str):
    """Generate embedding for text"""
    log.info("[Embedding] Generating embedding for text: %s", text[:80])
    resp = genai.embed_content(model="models/embedding-001", content=text)
    embedding = resp["embedding"]
    log.info("[Embedding] Embedding generated successfully (dim=%d)", len(embedding))
    return embedding

# ----------------------------
# Init Pinecone
# ----------------------------
pinecone_index = None
if PINECONE_API_KEY:
    pc = Pinecone(api_key=PINECONE_API_KEY)
    existing = [idx.name for idx in pc.list_indexes()]
    if PINECONE_INDEX not in existing:
        log.info("[Pinecone] Creating index: %s", PINECONE_INDEX)
        pc.create_index(
            name=PINECONE_INDEX,
            dimension=768,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-west-2")
        )
    pinecone_index = pc.Index(PINECONE_INDEX)
    log.info("[Pinecone] Connected to index: %s", PINECONE_INDEX)

def vector_db_upsert(vector_id, embedding, metadata):
    if pinecone_index:
        pinecone_index.upsert(
            vectors=[{"id": vector_id, "values": embedding, "metadata": metadata}],
            namespace="default"
        )
        log.info("[Pinecone] Upserted vector (id=%s)", vector_id)

def semantic_search(query: str, top_k: int = 3):
    if not pinecone_index:
        log.warning("[Pinecone] Semantic search skipped (index not initialized)")
        return []
    log.info("[SemanticSearch] Running semantic search for query: %s", query[:80])
    q_emb = get_embedding(query)
    resp = pinecone_index.query(vector=q_emb, top_k=top_k, include_metadata=True)
    matches = resp.get("matches", [])
    log.info("[SemanticSearch] Retrieved %d matches", len(matches))
    return matches

# ----------------------------
# Postgres
# ----------------------------
def get_db_connection():
    log.info("[DB] Connecting to PostgreSQL...")
    return psycopg2.connect(
        host=POSTGRES_HOST,
        database=POSTGRES_DB,
        user=POSTGRES_USER,
        password=POSTGRES_PASS,
        port=POSTGRES_PORT
    )

def log_message(conversation_id, message_id, sender, text, ts):
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(
            """INSERT INTO whatsapp_logs 
               (conversation_id, message_id, sender, message_text, created_at)
               VALUES (%s, %s, %s, %s, %s)""",
            (conversation_id, message_id, sender, text, datetime.fromtimestamp(int(ts)))
        )
        conn.commit()
        cur.close()
        conn.close()
        log.info("[DB] Logged message (id=%s, sender=%s)", message_id, sender)
    except Exception as e:
        log.exception("[DB] Failed to log message: %s", e)

# ----------------------------
# RAG Response
# ----------------------------
def get_ai_response(user_message: str, msg_id: str):
    log.info("[AI] Processing user message: %s", user_message[:80])

    # Upsert user query into Pinecone
    emb = get_embedding(user_message)
    print(">>>>>>>>>>>>>>>>>emb>>>>>>>>>>>>>>>>>>>>>>>>",emb)
    vector_db_upsert(msg_id, emb, {"text": user_message})

    # Retrieve context
    matches = semantic_search(user_message)
    print(">>>>>>>>>>>>>>>>>matches>>>>>>>>>>>>>>>>>>>>>>>>",matches)
    context = "\n".join([m["metadata"].get("text", "") for m in matches]) or "(no context found)"
    print(">>>>>>>>>>>>>>>>>context>>>>>>>>>>>>>>>>>>>>>>>>",context)
    log.info("[AI] Context prepared with %d chunks", len(matches))

    # Generate reply
    prompt = f"Context:\n{context}\n\nUser: {user_message}\n\nAnswer concisely."
    resp = gemini_model.generate_content(prompt)
    ai_text = resp.text if hasattr(resp, "text") else str(resp)
    log.info("[AI] Generated reply: %s", ai_text[:80])
    return ai_text

# ----------------------------
# WATI Helper
# ----------------------------
def send_whatsapp_message(phone_number: str, message: str):
    log.info("[WATI] Sending message to %s", phone_number)
    encoded = quote_plus(message)
    url = f"{BASE_URL}/api/v1/sendSessionMessage/{phone_number}?messageText={encoded}"
    headers = {"Authorization": f"Bearer {WATI_API_KEY}"}
    r = requests.post(url, headers=headers, timeout=10)
    if r.status_code == 200:
        log.info("[WATI] Message sent successfully")
        return r.json()
    else:
        log.error("[WATI] Failed to send message (status=%s): %s", r.status_code, r.text)
        return {"error": r.text}

# ----------------------------
# Flask App
# ----------------------------
app = Flask(__name__)

@app.route("/wati-webhook", methods=["POST"])
def wati_webhook():
    data = request.get_json(force=True, silent=True) or {}
    messages = data.get("messages", {}).get("items", [data])

    responses = []
    for msg in messages:
        text = msg.get("text") or msg.get("body")
        msg_id = msg.get("id") or str(datetime.utcnow().timestamp())
        conversation_id = msg.get("conversationId", "conv-default")
        timestamp = msg.get("timestamp") or int(datetime.utcnow().timestamp())
        from_number = msg.get("from") or os.getenv("TEST_RECIPIENT_PHONE")

        if text:
            log.info("[Webhook] Received message: %s", text)

            # 1. Log user message
            log_message(conversation_id, msg_id, "user", text, timestamp)

            # 2. Get AI response
            ai_reply = get_ai_response(text, msg_id)

            # 3. Log bot reply
            log_message(conversation_id, f"bot-{msg_id}", "bot", ai_reply, timestamp)

            # 4. Send reply
            wati_resp = send_whatsapp_message(from_number, ai_reply)

            responses.append({"user": text, "bot": ai_reply, "wati": wati_resp})

    return jsonify(responses)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "time": datetime.utcnow().isoformat()})

if __name__ == "__main__":
    log.info("🚀 Starting Flask server on port 5000...")
    app.run(port=5000, debug=True)