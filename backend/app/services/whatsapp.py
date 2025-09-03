# from flask import current_app
# import requests, os

# def send_whatsapp_text(to: str, body: str):
#     """
#     Twilio sandbox-friendly send function.
#     You can replace with Meta Cloud API if needed.
#     """
#     sid = os.getenv("TWILIO_ACCOUNT_SID")
#     token = os.getenv("TWILIO_AUTH_TOKEN")
#     from_ = os.getenv("WHATSAPP_FROM", "whatsapp:+14155238886")
#     url = f"https://api.twilio.com/2010-04-01/Accounts/{sid}/Messages.json"
#     data = {
#         "From": from_,
#         "To": to,
#         "Body": body
#     }
#     r = requests.post(url, data=data, auth=(sid, token), timeout=30)
#     if r.status_code >= 300:
#         current_app.logger.error("Twilio send error %s %s", r.status_code, r.text)
#     return r.ok
