# # import pinecone
# # from pinecone import (
# #     Pinecone,
# #     ServerlessSpec,
# #     CloudProvider,
# #     AwsRegion,
# #     VectorType
# # )

# # import numpy as np
# # import requests

# # PINECONE_API_KEY= "pcsk_5ZJEVn_K6FrjVje2XZnYuqxyhfJVYDVKuKg5A6RZc4UWaPKNzARdQxKK82o2xNc82paxBk"
# # # PINECONE_ENV="us-west1-gcp"
# # PINECONE_ENV = "us-east-1"
# # PINECONE_INDEX="nisaa-knowledge"
# # NAMESPACE="hospital"
# # GOOGLE_API_KEY="AIzaSyBT7ulr-_i-O1Z42rKLDr8ZJJjh9v52StM"
# # GEMINI_MODEL="gemini-1.5-pro"
# # # EMBEDDING_MODEL="models/embedding-001"
# # EMBEDDING_MODEL="models/llama-text-embed-v2"


# # # Initialize Pinecone
# # pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)

# # # Check available indexes to ensure correct setup
# # indexes = pc.list_indexes()
# # print(f"Available Indexes: {indexes}")

# # # Connect to your Pinecone index
# # index = pc.Index(PINECONE_INDEX)

# # # Step 1: Generate embedding from GenAI (Gemini API)
# # query_text = "give me all doctors list"
# # # genai_url = "https://api.gemini.google.com/v1/embed/text"

# # # genai_url = f"https://generativelanguage.googleapis.com/v1beta2/{EMBEDDING_MODEL}:embedText"
# # genai_url = f"https://generativelanguage.googleapis.com/v1beta2/{EMBEDDING_MODEL}:embedText"


# # headers = {
# #     "Content-Type": "application/json"
# # }

# # payload = {
# #     "text": query_text
# # }

# # params = {
# #     "key": GOOGLE_API_KEY
# # }

# # response = requests.post(genai_url, headers=headers, json=payload, params=params)
# # response.raise_for_status()

# # embedding_data = response.json()
# # query_vector = embedding_data['embedding']  # Correct key path for embedding result

# # # Step 2: Query Pinecone Vector DB
# # query_response = index.query(
# #     vector=query_vector,
# #     top_k=5,
# #     namespace=NAMESPACE,
# #     include_metadata=True
# # )

# # # Print results
# # print("Query Results:")
# # for match in query_response['matches']:
# #     print(f"ID: {match['id']}, Score: {match['score']}, Metadata: {match.get('metadata')}")




# # import google.generativeai as genai
# # from pinecone import Pinecone
# # import os




# # # ------------------------------
# # # Initialize GenAI SDK
# # # ------------------------------
# # genai.configure(api_key=GOOGLE_API_KEY)

# # def get_embedding(text: str):
# #     """
# #     Returns a list[float] embedding for `text` using the configured embedding model.
# #     """
# #     try:
# #         resp = genai.embed_content(model="models/embedding-001", content=text)
# #         # Different response shapes can occur; try common keys
# #         if isinstance(resp, dict):
# #             if "embedding" in resp:
# #                 return resp["embedding"]
# #             if "embeddings" in resp and resp["embeddings"]:
# #                 # sometimes returns list of embeddings
# #                 return resp["embeddings"][0]
# #         # If resp is an object with attributes
# #         if hasattr(resp, "embedding"):
# #             return resp.embedding
# #         # fallback
# #         raise ValueError(f"No embedding found in response:+{resp}")
# #     except Exception as e:
# #         raise

# # # ------------------------------
# # # Step 1: Generate Embedding
# # # ------------------------------
# # query_text = "give me all doctors list"

# # # model = genai.GenerativeModel(EMBEDDING_MODEL)

# # # response = model.t .embed_text(query_text)
# # query_vector = get_embedding(query_text)  # embedding vector list
# # # ------------------------------
# # # Step 2: Initialize Pinecone
# # # ------------------------------
# # pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)

# # # Check available indexes
# # indexes = pc.list_indexes()
# # print(f"Available Indexes: {indexes}")

# # # Connect to Pinecone index
# # index = pc.Index(PINECONE_INDEX)

# # # ------------------------------
# # # Step 3: Query Pinecone
# # # ------------------------------
# # query_response = index.query(
# #     vector=query_vector,
# #     top_k=5,
# #     namespace=NAMESPACE,
# #     include_metadata=True
# # )

# # # ------------------------------
# # # Step 4: Print results
# # # ------------------------------
# # print("Query Results:")
# # for match in query_response['matches']:
# #     print(f"ID: {match['id']}, Score: {match['score']}, Metadata: {match.get('metadata')}")









# # import requests
# # import json
# from urllib.parse import unquote

# # API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJqdGkiOiIyMGUzNDhmMC00YTQ2LTRmMmMtYTEzYS1iODIwZjVjMWVjZTciLCJ1bmlxdWVfbmFtZSI6InJhaXNpbmcxMDB4QGdtYWlsLmNvbSIsIm5hbWVpZCI6InJhaXNpbmcxMDB4QGdtYWlsLmNvbSIsImVtYWlsIjoicmFpc2luZzEwMHhAZ21haWwuY29tIiwiYXV0aF90aW1lIjoiMDkvMDkvMjAyNSAxMzozMjo0MCIsInRlbmFudF9pZCI6IjQ4MjMxMyIsImRiX25hbWUiOiJtdC1wcm9kLVRlbmFudHMiLCJodHRwOi8vc2NoZW1hcy5taWNyb3NvZnQuY29tL3dzLzIwMDgvMDYvaWRlbnRpdHkvY2xhaW1zL3JvbGUiOiJBRE1JTklTVFJBVE9SIiwiZXhwIjoyNTM0MDIzMDA4MDAsImlzcyI6IkNsYXJlX0FJIiwiYXVkIjoiQ2xhcmVfQUkifQ.iCxpiexmTahp-kd7aHZiiDThHp4wQDtBOiuz7ToNgg8"
# # # BASE_URL = "https://live-server.wati.io"
# # BASE_URL= "https://live-mt-server.wati.io/482313"

# # phone_number = "918240651574"

# # encoded_message = unquote("Hello from WATI API!")

# # url = f"{BASE_URL}/api/v1/sendTemplateMessage/{phone_number}?messageText={encoded_message}"

# # headers = {
# #         "accept": "*/*",
# #         "Authorization": f"Bearer {API_KEY}"
# #     }

# # response = requests.post(url, headers=headers)
# # print(response.status_code)
# # print(response.json())


# # import requests
# # import json

# # url = "https://app-server.wati.io/api/v2/sendTemplateMessage?whatsappNumber=919330010183"

# # payload = json.dumps({
# #   "template_name": "welcome_wati_v2",
# #   "broadcast_name": "welcome_wati_v2",
# #   "parameters": [
# #     {
# #       "name": "name",
# #       "value": "Hi I am testing my api"
# #     }
# #   ]
# # })
# # headers = {
# #   'accept': '*/*',
# #   'Authorization': 'Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJqdGkiOiJhNGE2NzE0ZC02OTFiLTQxMDgtYTA2NC05NjRmMDQ1M2RmZGQiLCJ1bmlxdWVfbmFtZSI6ImFrYXNoLm11a2hlcmplZUBuem1pbmRzLmNvbSIsIm5hbWVpZCI6ImFrYXNoLm11a2hlcmplZUBuem1pbmRzLmNvbSIsImVtYWlsIjoiYWthc2gubXVraGVyamVlQG56bWluZHMuY29tIiwiYXV0aF90aW1lIjoiMDkvMDQvMjAyNSAwODoxMzo0MCIsImRiX25hbWUiOiJ3YXRpX2FwcF90cmlhbCIsImh0dHA6Ly9zY2hlbWFzLm1pY3Jvc29mdC5jb20vd3MvMjAwOC8wNi9pZGVudGl0eS9jbGFpbXMvcm9sZSI6IlRSSUFMIiwiZXhwIjoxNzU3NjM1MjAwLCJpc3MiOiJDbGFyZV9BSSIsImF1ZCI6IkNsYXJlX0FJIn0.A7DIELysmv6k0XRi7JNxRms2--dzW7ijiPa8WclE6Vc',
# #   'Content-Type': 'application/json-patch+json'
# # }

# # response = requests.request("POST", url, headers=headers, data=payload)

# # print(response.text)



# # import requests
# # import json

# # url = "https://live-mt-server.wati.io/482313/api/v2/sendTemplateMessage?whatsappNumber=919330010183"

# # payload = json.dumps({
# #   "template_name": "welcome_wati_v2",
# #   "broadcast_name": "welcome_wati_v2",
# #   "parameters": [
# #     {
# #       "name": "name",
# #       "value": "Hi I am testing Wati api"
# #     }
# #   ]
# # })
# # headers = {
# #   'accept': '*/*',
# #   'Authorization': 'Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJqdGkiOiIyMGUzNDhmMC00YTQ2LTRmMmMtYTEzYS1iODIwZjVjMWVjZTciLCJ1bmlxdWVfbmFtZSI6InJhaXNpbmcxMDB4QGdtYWlsLmNvbSIsIm5hbWVpZCI6InJhaXNpbmcxMDB4QGdtYWlsLmNvbSIsImVtYWlsIjoicmFpc2luZzEwMHhAZ21haWwuY29tIiwiYXV0aF90aW1lIjoiMDkvMDkvMjAyNSAxMzozMjo0MCIsInRlbmFudF9pZCI6IjQ4MjMxMyIsImRiX25hbWUiOiJtdC1wcm9kLVRlbmFudHMiLCJodHRwOi8vc2NoZW1hcy5taWNyb3NvZnQuY29tL3dzLzIwMDgvMDYvaWRlbnRpdHkvY2xhaW1zL3JvbGUiOiJBRE1JTklTVFJBVE9SIiwiZXhwIjoyNTM0MDIzMDA4MDAsImlzcyI6IkNsYXJlX0FJIiwiYXVkIjoiQ2xhcmVfQUkifQ.iCxpiexmTahp-kd7aHZiiDThHp4wQDtBOiuz7ToNgg8',
# #   'Content-Type': 'application/json-patch+json'
# # }

# # response = requests.request("POST", url, headers=headers, data=payload)

# # print(response.text)



# import requests
# import json

# API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJqdGkiOiIyMGUzNDhmMC00YTQ2LTRmMmMtYTEzYS1iODIwZjVjMWVjZTciLCJ1bmlxdWVfbmFtZSI6InJhaXNpbmcxMDB4QGdtYWlsLmNvbSIsIm5hbWVpZCI6InJhaXNpbmcxMDB4QGdtYWlsLmNvbSIsImVtYWlsIjoicmFpc2luZzEwMHhAZ21haWwuY29tIiwiYXV0aF90aW1lIjoiMDkvMDkvMjAyNSAxMzozMjo0MCIsInRlbmFudF9pZCI6IjQ4MjMxMyIsImRiX25hbWUiOiJtdC1wcm9kLVRlbmFudHMiLCJodHRwOi8vc2NoZW1hcy5taWNyb3NvZnQuY29tL3dzLzIwMDgvMDYvaWRlbnRpdHkvY2xhaW1zL3JvbGUiOiJBRE1JTklTVFJBVE9SIiwiZXhwIjoyNTM0MDIzMDA4MDAsImlzcyI6IkNsYXJlX0FJIiwiYXVkIjoiQ2xhcmVfQUkifQ.iCxpiexmTahp-kd7aHZiiDThHp4wQDtBOiuz7ToNgg8"
# BASE_URL = "https://live-mt-server.wati.io/482313"

# whatsapp_number = "919330010183"
# template_name = "vrr_realestate"  # Must match exactly
# language_code = "en"  # For example

# # url = f"{BASE_URL}/api/v1/sendTemplateMessage"

# # headers = {
# #     "Authorization": f"Bearer {API_KEY}",
# #     "Content-Type": "application/json"
# # }
# # payload = {
# #     "template_name": template_name,   # EXACT template name from WATI Dashboard
# #     "broadcast_name": "MyBroadcast",         # Any meaningful broadcast name
# #     "parameters": [
# #         {"name": "param1", "value": "Testing"},   # Match your template placeholder names
# #         {"name": "param2", "value": "Api"}
# #     ],
# #     "channel_number": whatsapp_number         # Your WhatsApp business channel number
# # }


# # response = requests.post(url, headers=headers, data=json.dumps(payload))
# # print(response.status_code)
# # print(response.json())


# # phone_number = "919330010183"

# # encoded_message = unquote("Hello from WATI API!")

# # url = f"{BASE_URL}/api/v1/sendTemplateMessage/{phone_number}?messageText={encoded_message}"

# # headers = {
# #         "accept": "*/*",
# #         "Authorization": f"Bearer {API_KEY}"
# #     }

# # response = requests.post(url, headers=headers)
# # print(response.status_code)
# # print(response.json())




# import requests
# import json
# from urllib.parse import unquote


# TENANT_ID = "482313"
# ACCESS_TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJqdGkiOiIyMGUzNDhmMC00YTQ2LTRmMmMtYTEzYS1iODIwZjVjMWVjZTciLCJ1bmlxdWVfbmFtZSI6InJhaXNpbmcxMDB4QGdtYWlsLmNvbSIsIm5hbWVpZCI6InJhaXNpbmcxMDB4QGdtYWlsLmNvbSIsImVtYWlsIjoicmFpc2luZzEwMHhAZ21haWwuY29tIiwiYXV0aF90aW1lIjoiMDkvMDkvMjAyNSAxMzozMjo0MCIsInRlbmFudF9pZCI6IjQ4MjMxMyIsImRiX25hbWUiOiJtdC1wcm9kLVRlbmFudHMiLCJodHRwOi8vc2NoZW1hcy5taWNyb3NvZnQuY29tL3dzLzIwMDgvMDYvaWRlbnRpdHkvY2xhaW1zL3JvbGUiOiJBRE1JTklTVFJBVE9SIiwiZXhwIjoyNTM0MDIzMDA4MDAsImlzcyI6IkNsYXJlX0FJIiwiYXVkIjoiQ2xhcmVfQUkifQ.iCxpiexmTahp-kd7aHZiiDThHp4wQDtBOiuz7ToNgg8"
# # BASE_URL = "https://live-mt-server.wati.io/482313"


# BASE_URL = f"https://live-mt-server.wati.io/{TENANT_ID}/api/v1"

# # HEADERS = {
# #     "Authorization": f"Bearer {ACCESS_TOKEN}",
# #     "Content-Type": "application/json"
# # }


# # def get_contacts():
# #     url = f"{BASE_URL}/getContacts"
# #     response = requests.get(url, headers=HEADERS)
# #     print(">>>>>>>>esponse.status_code>>>>>>>>>",response.status_code)
# #     print(">>>>>>>>>esponse,json>>>>>>>",response.json())
# #     return response.json()

# # # get_contacts()

# # def send_session_message(whatsapp_number, message):
# #     if not message or message.strip() == "":
# #         raise ValueError("Message text cannot be empty.")
# #     encoded_message = unquote(message)
# #     url = f"{BASE_URL}/sendSessionMessage/{whatsapp_number}?messageText={encoded_message}"
# #     # payload = {
# #     #     "messageText": message.strip()
# #     # }

# #     # url = f"{BASE_URL}/api/v1/sendSessionMessage/{phone_number}?messageText={encoded_message}"


# #     # headers = {
# #     #     "accept": "*/*",
# #     #     "Authorization": f"Bearer {ACCESS_TOKEN}"
# #     # }
# #     headers = {
# #         'Content-Type': 'application/json',
# #         'x-wa-api-key': ACCESS_TOKEN,   # your API key
# #     }

# #     response = requests.post(url, headers=headers)
# #     return response.json()
# # try:
# #     result = send_session_message("919669092627", "Hello, this is a test message!")
# #     print("Send Session Message Response:", result)
# # except ValueError as e:
# #     print(f"Error: {e}")




# import requests
# import json

# # Configuration
# API_KEY = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJqdGkiOiIyMGUzNDhmMC00YTQ2LTRmMmMtYTEzYS1iODIwZjVjMWVjZTciLCJ1bmlxdWVfbmFtZSI6InJhaXNpbmcxMDB4QGdtYWlsLmNvbSIsIm5hbWVpZCI6InJhaXNpbmcxMDB4QGdtYWlsLmNvbSIsImVtYWlsIjoicmFpc2luZzEwMHhAZ21haWwuY29tIiwiYXV0aF90aW1lIjoiMDkvMDkvMjAyNSAxMzozMjo0MCIsInRlbmFudF9pZCI6IjQ4MjMxMyIsImRiX25hbWUiOiJtdC1wcm9kLVRlbmFudHMiLCJodHRwOi8vc2NoZW1hcy5taWNyb3NvZnQuY29tL3dzLzIwMDgvMDYvaWRlbnRpdHkvY2xhaW1zL3JvbGUiOiJBRE1JTklTVFJBVE9SIiwiZXhwIjoyNTM0MDIzMDA4MDAsImlzcyI6IkNsYXJlX0FJIiwiYXVkIjoiQ2xhcmVfQUkifQ.iCxpiexmTahp-kd7aHZiiDThHp4wQDtBOiuz7ToNgg8'        # Replace with your actual API key
# TENANT_ID = '482313'        # Replace with your actual tenant ID
# PHONE_NUMBER = '919669092627'       # Target WhatsApp number with country code
# MESSAGE_TEXT = 'Hello from WATI API!'



# import requests
# import json

# # def send_whatsapp_message(api_key, tenant_id, phone_number, message_text):
# #     # BASE_URL = 'https://live-server.wati.io/api/v1/sendSessionMessage'
# #     BASE_URL = 'https://live-mt-server.wati.io/api/v1/sendSessionMessage'

# #     headers = {
# #         'Content-Type': 'application/json',
# #         'x-wa-api-key': api_key
# #     }

# #     payload = {
# #         "tenantId": tenant_id,
# #         "to": phone_number,
# #         "message": {
# #             "type": "text",
# #             "text": message_text
# #         }
# #     }

# #     try:
# #         response = requests.post(BASE_URL, headers=headers, data=json.dumps(payload), timeout=10)

# #         print(f"HTTP Status Code: {response.status_code}")

# #         # Ensure JSON response, else print text for debug
# #         try:
# #             response_data = response.json()
# #             return response_data
# #         except json.decoder.JSONDecodeError:
# #             print("Response is not valid JSON. Raw response:")
# #             print(response.text)
# #             return {"error": "Invalid JSON response", "response_text": response.text}

# #     except requests.exceptions.RequestException as e:
# #         return {"error": "Request failed", "exception": str(e)}


# # # Example Usage
# # if __name__ == "__main__":

# #     result = send_whatsapp_message(API_KEY, TENANT_ID, PHONE_NUMBER, MESSAGE_TEXT)
# #     print(json.dumps(result, indent=4))





# # import requests
# # import json

# # # === Configuration ===
# # # BASE_URL = "https://live-server.wati.io/api/v1"
# # BASE_URL = "https://live-mt-server.wati.io"

# # HEADERS = {
# #     "Authorization": f"Bearer {API_KEY}",
# #     "Content-Type": "application/json"
# # }

# # def send_whatsapp_message(phone_number, message_text):
# #     url = f"{BASE_URL}/api/v1/sendSessionMessage"
# #     payload = {
# #         "tenantId": TENANT_ID,
# #         "to": phone_number,              # Example: "919876543210"
# #         "message": message_text          # Example: "Hello from WATI API"
# #     }

# #     response = requests.post(url, headers=HEADERS, data=json.dumps(payload))
# #     print(">>>>>>>>>>>>>>>>>",response.status_code)
# #     print(">>>>>>>>>>>>",response.json)
# #     try:
# #         response_json = response.json()
# #     except json.JSONDecodeError:
# #         print(f"Invalid JSON response: {response.text}")
# #         return None

# #     return response_json


# # # === Example Usage ===
# # if __name__ == "__main__":
# #     phone = PHONE_NUMBER  # International format without "+" or spaces
# #     message = "Hello, this is a test message from WATI API."

# #     result = send_whatsapp_message(phone, message)
# #     print("Send Message Response:", json.dumps(result, indent=2))




# import requests

# url = "https://live-np-server.wati.io/124785/api/v1/sendSessionMessage/91092627?messageText=Hi%20This%20is%20for%20APi%20Testing&channelPhoneNumber=917337320100"

# payload = {}
# headers = {
# 'accept': '*/*',
# 'Authorization': 'Bearer eyJhbGcig8'
# }

# response = requests.request("POST", url, headers=headers, data=payload)

# print(response.text)


# import os
# import requests

# Load environment variables
# API_KEY = os.getenv("API_KEY")
# TENANT_ID = os.getenv("TENANT_ID")
# PHONE_NUMBER = os.getenv("PHONE_NUMBER")
# CHANNEL_NUMBER = os.getenv("CHANNEL_NUMBER")



# API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJqdGkiOiIyMGUzNDhmMC00YTQ2LTRmMmMtYTEzYS1iODIwZjVjMWVjZTciLCJ1bmlxdWVfbmFtZSI6InJhaXNpbmcxMDB4QGdtYWlsLmNvbSIsIm5hbWVpZCI6InJhaXNpbmcxMDB4QGdtYWlsLmNvbSIsImVtYWlsIjoicmFpc2luZzEwMHhAZ21haWwuY29tIiwiYXV0aF90aW1lIjoiMDkvMDkvMjAyNSAxMzozMjo0MCIsInRlbmFudF9pZCI6IjQ4MjMxMyIsImRiX25hbWUiOiJtdC1wcm9kLVRlbmFudHMiLCJodHRwOi8vc2NoZW1hcy5taWNyb3NvZnQuY29tL3dzLzIwMDgvMDYvaWRlbnRpdHkvY2xhaW1zL3JvbGUiOiJBRE1JTklTVFJBVE9SIiwiZXhwIjoyNTM0MDIzMDA4MDAsImlzcyI6IkNsYXJlX0FJIiwiYXVkIjoiQ2xhcmVfQUkifQ.iCxpiexmTahp-kd7aHZiiDThHp4wQDtBOiuz7ToNgg8"
# TENANT_ID = "482313"
# PHONE_NUMBER = "919669092627"
# CHANNEL_NUMBER = "917337320100"
# BASE_URL="https://live-mt-server.wati.io"

# # Construct URL using TENANT_ID and PHONE_NUMBER
# url = f"{BASE_URL}/{TENANT_ID}/api/v1/sendSessionMessage/{PHONE_NUMBER}"
# params = {
#     "messageText": "Hi Tanmay how are you",
#     "channelPhoneNumber": CHANNEL_NUMBER
# }

# headers = {
#     'accept': '*/*',
#     'Authorization': f'Bearer {API_KEY}'
# }

# response = requests.post(url, headers=headers, params=params)
# print(response.status_code)
# print(response.text)




# import requests

# url = "https://live-mt-server.wati.io/482313/api/v1/getMessages/919669092627?channelPhoneNumber=917337320100"

# payload = {}
# headers = {
#   'accept': '*/*',
#   'Authorization': 'Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJqdGkiOiIyMGUzNDhmMC00YTQ2LTRmMmMtYTEzYS1iODIwZjVjMWVjZTciLCJ1bmlxdWVfbmFtZSI6InJhaXNpbmcxMDB4QGdtYWlsLmNvbSIsIm5hbWVpZCI6InJhaXNpbmcxMDB4QGdtYWlsLmNvbSIsImVtYWlsIjoicmFpc2luZzEwMHhAZ21haWwuY29tIiwiYXV0aF90aW1lIjoiMDkvMDkvMjAyNSAxMzozMjo0MCIsInRlbmFudF9pZCI6IjQ4MjMxMyIsImRiX25hbWUiOiJtdC1wcm9kLVRlbmFudHMiLCJodHRwOi8vc2NoZW1hcy5taWNyb3NvZnQuY29tL3dzLzIwMDgvMDYvaWRlbnRpdHkvY2xhaW1zL3JvbGUiOiJBRE1JTklTVFJBVE9SIiwiZXhwIjoyNTM0MDIzMDA4MDAsImlzcyI6IkNsYXJlX0FJIiwiYXVkIjoiQ2xhcmVfQUkifQ.iCxpiexmTahp-kd7aHZiiDThHp4wQDtBOiuz7ToNgg8'
# }

# response = requests.request("GET", url, headers=headers, data=payload)

# print(response.text)
