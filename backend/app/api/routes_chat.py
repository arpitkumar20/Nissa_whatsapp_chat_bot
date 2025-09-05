import os
from flask import Blueprint, request, jsonify
from app.services.wati_service import send_whatsapp_message , get_whatsapp_messages
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

        if not sender_number:
            return jsonify({"error": "Missing required fields (sender_number)"}), 400

        result = get_whatsapp_messages(sender_number)
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500
