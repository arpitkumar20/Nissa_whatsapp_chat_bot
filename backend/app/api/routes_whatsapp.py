import os
from urllib.parse import unquote
import requests
from flask import Flask, request, jsonify
from datetime import datetime
import psycopg2
from psycopg2.extras import RealDictCursor
import google.generativeai as genai
# ----------------------------
# CONFIG
# ----------------------------


BASE_URL = "https://app-server.wati.io"
API_KEY = os.getenv('WATI_APY_KEY')
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")

# Init Gemini
genai.configure(api_key=GOOGLE_API_KEY)
gemini_model = genai.GenerativeModel("gemini-1.5-pro")
embedding_model = "models/embedding-001"

# Init Flask
app = Flask(__name__)

# ----------------------------
# PostgreSQL + PGVector Connection
# ----------------------------
def get_db_connection():
    print("[DB] Connecting to PostgreSQL...")
    conn = psycopg2.connect(
        host="localhost",
        database="chatdb",
        user="postgres",
        password="root"
    )
    print("[DB] Connection successful")
    return conn

def send_whatsapp_message(phone_number, message) -> dict:
    """
    Send a WhatsApp session message using the WATI API.
    """
    # URL encode the message
    encoded_message = unquote(message)

    # url = f"{BASE_URL}/sendSessionMessage/{phone_number}?messageText={encoded_message}"
    url = f"{BASE_URL}/api/v1/sendSessionMessage/{phone_number}?messageText={encoded_message}"

    headers = {
        "accept": "*/*",
        "Authorization": f"Bearer {API_KEY}"
    }

    try:
        response = requests.post(url, headers=headers)
        if response.status_code == 200:
            return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}

# ----------------------------
# Helper: Log messages in DB
# ----------------------------
def log_message(conversation_id, msg_id, sender, text, timestamp):
    print(f"[DB] Logging message | sender={sender}, text={text}")
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO whatsapp_logs (conversation_id, message_id, sender, message_text, created_at)
        VALUES (%s, %s, %s, %s, %s)
        """,
        (conversation_id, msg_id, sender, text, datetime.fromtimestamp(int(timestamp)))
    )
    conn.commit()
    cursor.close()
    conn.close()
    print("[DB] Message logged successfully")

# ----------------------------
# Embeddings
# ----------------------------
def get_embedding(text: str):
    print(f"[Embedding] Generating embedding for text: {text}")
    result = genai.embed_content(
        model=embedding_model,
        content=text
    )
    embedding = result["embedding"]
    print("[Embedding] Embedding generated")
    return embedding

import numpy as np

def cosine_similarity(vec1, vec2):
    v1 = np.array(vec1)
    v2 = np.array(vec2)
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def semantic_search(user_embedding, top_k=3):
    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    cursor.execute("SELECT id, chunk, embedding FROM knowledge_base;")
    rows = cursor.fetchall()
    cursor.close()
    conn.close()

    # Compute cosine similarity in Python
    for row in rows:
        row["similarity"] = cosine_similarity(user_embedding, row["embedding"])

    # Sort by similarity (descending)
    rows = sorted(rows, key=lambda r: r["similarity"], reverse=True)
    return rows[:top_k]

# ----------------------------
# AI Response (Gemini + Context)
# ----------------------------
def get_ai_response(user_message):
    print(f"[AI] User message: {user_message}")

    # 1. Embed user query
    user_embedding = get_embedding(user_message)

    # 2. Search KB
    top_chunks = semantic_search(user_embedding, top_k=3)
    context_text = "\n".join([f"(ID {row['id']}) {row['chunk']}" for row in top_chunks])
    retrieved_ids = [row["id"] for row in top_chunks]
    print(f"[AI] Retrieved context IDs: {retrieved_ids}")

    # 3. Build prompt
    prompt = f"""
    You are a helpful AI assistant.
    User asked: {user_message}

    Relevant knowledge base items:
    {context_text}

    Use the context to answer. If not enough info, say so.
    """
    print("[AI] Sending prompt to Gemini...")
    response = gemini_model.generate_content(prompt)
    print("[AI] Gemini response received")
    return response.text, retrieved_ids

# ----------------------------
# Flask route - Webhook (WATI)
# ----------------------------
@app.route("/wati-webhook", methods=["POST"])
def wati_webhook():
    data = request.json
    print("[Webhook] Incoming WATI Data:", data)

    messages = data.get("messages", {}).get("items", [])
    responses = []

    for msg in messages:
        text = msg.get("text")
        msg_type = msg.get("type")
        conversation_id = msg.get("conversationId")
        msg_id = msg.get("id")
        timestamp = msg.get("timestamp")
        owner = msg.get("owner")  # False = user, True = bot

        if msg_type == "text" and not owner and text:
            print(f"[Webhook] Processing user message: {text}")

            # 1. Log user message
            log_message(conversation_id, msg_id, "user", text, timestamp)

            # 2. AI response (RAG flow)
            ai_reply, retrieved_ids = get_ai_response(text)

            # 3. Log bot reply
            log_message(conversation_id, f"bot-{msg_id}", "bot", ai_reply, timestamp)

            # 4. Send reply to WhatsApp
            phone_number = "919669092627"  # Replace with actual phone number mapping
            send_resp = send_whatsapp_message(phone_number, ai_reply)

            responses.append({
                "user_message": text,
                "ai_reply": ai_reply,
                "retrieved_ids": retrieved_ids,
                "wati_response": send_resp
            })

    print("[Webhook] Responses prepared:", responses)
    return jsonify(responses)

# ----------------------------
# Run Flask
# ----------------------------
if __name__ == "__main__":
    print("[Server] Starting Flask server on port 5000...")
    app.run(port=5000, debug=True)
