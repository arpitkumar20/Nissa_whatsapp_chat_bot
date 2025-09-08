import os
import logging
from flask import Blueprint, request, jsonify
from app.services.wati_service import send_whatsapp_message , get_whatsapp_messages
from app.services.vectordb_retrive import query_pinecone
from app.services.genai_response import handle_user_query

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

chat_bp = Blueprint("chat", __name__)

BASE_URL = "https://app-server.wati.io"
API_KEY = os.getenv('WATI_APY_KEY')

@chat_bp.route("/ping", methods=["GET"])
def ping():
    return jsonify({"status": "ok", "message": "Chat API is alive!"})

@chat_bp.route("/send-message", methods=["POST"])
def send_message():
    data = request.json or {}

    phone_number = data.get("phone_number")
    message = data.get("message")

    if not phone_number or not message:
        return jsonify({"error": "Missing required fields (phone_number, message)"}), 400

    result = send_whatsapp_message(phone_number, message)
    return jsonify(result)


@chat_bp.route("/receive-message", methods=["POST"])
def receive_message():
    """
    Receive incoming WhatsApp messages via WATI webhook.
    """
    try:
        data = request.json  # WATI sends incoming message here
        if not data:
            return jsonify({"error": "Empty payload"}), 400

        # Extract sender number
        sender_number = data.get("sender_number")

        # if not sender_number:
        #     return jsonify({"error": "Missing required fields (sender_number)"}), 400

        # result = get_whatsapp_messages(sender_number)
        # last_user_message = [result["last_user_message"]["text"]] if result["last_user_message"] else []
        # print(">>>>>>>>>>>>last_user_message>>>>>>>>>>>>")
        # query_response = query_pinecone(last_user_message[0])
        # print(">>>>>>>>>>>>query_response>>>>>>>>>>>>",query_response)
        # exit(0)
        # genai_response = handle_user_query(query_response)
        # print(">>>>>>>>>>>>genai_response>>>>>>>>>>>>",genai_response)
        # app_result = send_whatsapp_message(phone_number="919669092627", message=genai_response.get('answer'))
        if not sender_number:
            logger.error("Missing required fields: sender_number")
            return jsonify({"error": "Missing required fields (sender_number)"}), 400

        logger.info("Fetching WhatsApp messages for sender_number: %s", sender_number)
        result = get_whatsapp_messages(sender_number)

        if result.get("last_user_message"):
            last_user_message = result["last_user_message"].get("text", "")
            logger.info("Last user message retrieved: %s", last_user_message)
        else:
            last_user_message = ""
            logger.warning("No last user message found for sender_number: %s", sender_number)

        if last_user_message:
            logger.info("Querying Pinecone with last_user_message")
            query_response = query_pinecone(last_user_message)
            
            logger.info("Generating GenAI response based on Pinecone query")
            genai_response = handle_user_query(query_response)
            
            logger.info("Sending WhatsApp message to user phone_number")
            app_result = send_whatsapp_message(
                phone_number="919669092627",
                message=genai_response.get('answer')
            )
            logger.info("WhatsApp message send result: %s", app_result)
        else:
            logger.warning("Skipping Pinecone query and GenAI response since last_user_message is empty")
        return jsonify(app_result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500
