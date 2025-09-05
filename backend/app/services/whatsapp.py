# import requests

# url = "https://app-server.wati.io/api/v1/getContacts"

# headers = {"Authorization": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJqdGkiOiJhNGE2NzE0ZC02OTFiLTQxMDgtYTA2NC05NjRmMDQ1M2RmZGQiLCJ1bmlxdWVfbmFtZSI6ImFrYXNoLm11a2hlcmplZUBuem1pbmRzLmNvbSIsIm5hbWVpZCI6ImFrYXNoLm11a2hlcmplZUBuem1pbmRzLmNvbSIsImVtYWlsIjoiYWthc2gubXVraGVyamVlQG56bWluZHMuY29tIiwiYXV0aF90aW1lIjoiMDkvMDQvMjAyNSAwODoxMzo0MCIsImRiX25hbWUiOiJ3YXRpX2FwcF90cmlhbCIsImh0dHA6Ly9zY2hlbWFzLm1pY3Jvc29mdC5jb20vd3MvMjAwOC8wNi9pZGVudGl0eS9jbGFpbXMvcm9sZSI6IlRSSUFMIiwiZXhwIjoxNzU3NjM1MjAwLCJpc3MiOiJDbGFyZV9BSSIsImF1ZCI6IkNsYXJlX0FJIn0.A7DIELysmv6k0XRi7JNxRms2--dzW7ijiPa8WclE6Vc"}

# response = requests.get(url, headers=headers)

# print(response.text)

# '''
# {"result":"success","login_user_phone":"918240651574"}

# '''



# import requests

# url = "https://app-server.wati.io/api/v1/getContacts"
# url = "https://app-server.wati.io/api/v1/getMessages/+918240651574"


# headers = {"Authorization": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJqdGkiOiJhNGE2NzE0ZC02OTFiLTQxMDgtYTA2NC05NjRmMDQ1M2RmZGQiLCJ1bmlxdWVfbmFtZSI6ImFrYXNoLm11a2hlcmplZUBuem1pbmRzLmNvbSIsIm5hbWVpZCI6ImFrYXNoLm11a2hlcmplZUBuem1pbmRzLmNvbSIsImVtYWlsIjoiYWthc2gubXVraGVyamVlQG56bWluZHMuY29tIiwiYXV0aF90aW1lIjoiMDkvMDQvMjAyNSAwODoxMzo0MCIsImRiX25hbWUiOiJ3YXRpX2FwcF90cmlhbCIsImh0dHA6Ly9zY2hlbWFzLm1pY3Jvc29mdC5jb20vd3MvMjAwOC8wNi9pZGVudGl0eS9jbGFpbXMvcm9sZSI6IlRSSUFMIiwiZXhwIjoxNzU3NjM1MjAwLCJpc3MiOiJDbGFyZV9BSSIsImF1ZCI6IkNsYXJlX0FJIn0.A7DIELysmv6k0XRi7JNxRms2--dzW7ijiPa8WclE6Vc"}

# response = requests.get(url, headers=headers)

# print(response.text)


# import requests
# import json

# url = "https://app-server.wati.io/api/v2/sendTemplateMessage?whatsappNumber=919470018980"

# payload = json.dumps({
#   "template_name": "welcome_wati_v2",
#   "broadcast_name": "welcome_wati_v2",
#   "parameters": [
#     {
#       "name": "name",
#       "value": "Akash"
#     }
#   ]
# })
# headers = {
#   'accept': '*/*',
#   'Authorization': 'Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJqdGkiOiJhNGE2NzE0ZC02OTFiLTQxMDgtYTA2NC05NjRmMDQ1M2RmZGQiLCJ1bmlxdWVfbmFtZSI6ImFrYXNoLm11a2hlcmplZUBuem1pbmRzLmNvbSIsIm5hbWVpZCI6ImFrYXNoLm11a2hlcmplZUBuem1pbmRzLmNvbSIsImVtYWlsIjoiYWthc2gubXVraGVyamVlQG56bWluZHMuY29tIiwiYXV0aF90aW1lIjoiMDkvMDQvMjAyNSAwODoxMzo0MCIsImRiX25hbWUiOiJ3YXRpX2FwcF90cmlhbCIsImh0dHA6Ly9zY2hlbWFzLm1pY3Jvc29mdC5jb20vd3MvMjAwOC8wNi9pZGVudGl0eS9jbGFpbXMvcm9sZSI6IlRSSUFMIiwiZXhwIjoxNzU3NjM1MjAwLCJpc3MiOiJDbGFyZV9BSSIsImF1ZCI6IkNsYXJlX0FJIn0.A7DIELysmv6k0XRi7JNxRms2--dzW7ijiPa8WclE6Vc',
#   'Content-Type': 'application/json-patch+json',
#   'Cookie': 'affinity=1756977189.011.39.411364|aab056cd4cc9b597f01aa146c61e0719'
# }

# response = requests.request("POST", url, headers=headers, data=payload)

# print(response.text)




# import requests

# url = "https://app-server.wati.io/api/v1/sendSessionMessage/919470018980?messageText=Hi%20my%20name%20is%20Akash%20How%20are%20you"

# payload = {}
# headers = {
#   'accept': '*/*',
#   'Authorization': 'Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJqdGkiOiJhNGE2NzE0ZC02OTFiLTQxMDgtYTA2NC05NjRmMDQ1M2RmZGQiLCJ1bmlxdWVfbmFtZSI6ImFrYXNoLm11a2hlcmplZUBuem1pbmRzLmNvbSIsIm5hbWVpZCI6ImFrYXNoLm11a2hlcmplZUBuem1pbmRzLmNvbSIsImVtYWlsIjoiYWthc2gubXVraGVyamVlQG56bWluZHMuY29tIiwiYXV0aF90aW1lIjoiMDkvMDQvMjAyNSAwODoxMzo0MCIsImRiX25hbWUiOiJ3YXRpX2FwcF90cmlhbCIsImh0dHA6Ly9zY2hlbWFzLm1pY3Jvc29mdC5jb20vd3MvMjAwOC8wNi9pZGVudGl0eS9jbGFpbXMvcm9sZSI6IlRSSUFMIiwiZXhwIjoxNzU3NjM1MjAwLCJpc3MiOiJDbGFyZV9BSSIsImF1ZCI6IkNsYXJlX0FJIn0.A7DIELysmv6k0XRi7JNxRms2--dzW7ijiPa8WclE6Vc',
#   'Cookie': 'affinity=1756977189.011.39.411364|aab056cd4cc9b597f01aa146c61e0719'
# }

# response = requests.request("POST", url, headers=headers, data=payload)

# print(response.text)


# import requests

# url = "https://app-server.wati.io/api/v1/getMessages/919470018980?pageSize=1&pageNumber=1"

# payload = {}
# headers = {
#   'accept': '*/*',
#   'Authorization': 'Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJqdGkiOiJhNGE2NzE0ZC02OTFiLTQxMDgtYTA2NC05NjRmMDQ1M2RmZGQiLCJ1bmlxdWVfbmFtZSI6ImFrYXNoLm11a2hlcmplZUBuem1pbmRzLmNvbSIsIm5hbWVpZCI6ImFrYXNoLm11a2hlcmplZUBuem1pbmRzLmNvbSIsImVtYWlsIjoiYWthc2gubXVraGVyamVlQG56bWluZHMuY29tIiwiYXV0aF90aW1lIjoiMDkvMDQvMjAyNSAwODoxMzo0MCIsImRiX25hbWUiOiJ3YXRpX2FwcF90cmlhbCIsImh0dHA6Ly9zY2hlbWFzLm1pY3Jvc29mdC5jb20vd3MvMjAwOC8wNi9pZGVudGl0eS9jbGFpbXMvcm9sZSI6IlRSSUFMIiwiZXhwIjoxNzU3NjM1MjAwLCJpc3MiOiJDbGFyZV9BSSIsImF1ZCI6IkNsYXJlX0FJIn0.A7DIELysmv6k0XRi7JNxRms2--dzW7ijiPa8WclE6Vc',
#   'Cookie': 'affinity=1756977189.011.39.411364|aab056cd4cc9b597f01aa146c61e0719'
# }

# response = requests.request("GET", url, headers=headers, data=payload)

# print(response.text)


# import requests

# url = "https://app-server.wati.io/api/v1/sendSessionMessage/919470018980?pageSize=1&pageNumber=1"
# headers = {
#     "Authorization": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJqdGkiOiJhNGE2NzE0ZC02OTFiLTQxMDgtYTA2NC05NjRmMDQ1M2RmZGQiLCJ1bmlxdWVfbmFtZSI6ImFrYXNoLm11a2hlcmplZUBuem1pbmRzLmNvbSIsIm5hbWVpZCI6ImFrYXNoLm11a2hlcmplZUBuem1pbmRzLmNvbSIsImVtYWlsIjoiYWthc2gubXVraGVyamVlQG56bWluZHMuY29tIiwiYXV0aF90aW1lIjoiMDkvMDQvMjAyNSAwODoxMzo0MCIsImRiX25hbWUiOiJ3YXRpX2FwcF90cmlhbCIsImh0dHA6Ly9zY2hlbWFzLm1pY3Jvc29mdC5jb20vd3MvMjAwOC8wNi9pZGVudGl0eS9jbGFpbXMvcm9sZSI6IlRSSUFMIiwiZXhwIjoxNzU3NjM1MjAwLCJpc3MiOiJDbGFyZV9BSSIsImF1ZCI6IkNsYXJlX0FJIn0.A7DIELysmv6k0XRi7JNxRms2--dzW7ijiPa8WclE6Vc",
#     'accept': '*/*'
# }

# payload = {
#     "messageText": "🤖 [Nissa AI Bot]: Hello! How can I help you today?"
# }

# response = requests.post(url, headers=headers, json=payload)
# print(response.status_code)
# print(response.text)
# print(response.json())




# import requests

# url = "https://app-server.wati.io/api/v1/sendSessionMessage/919669092627?messageText=Hi%20testing"

# headers = {
#     "accept": "*/*",
#     "Authorization": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJqdGkiOiJhNGE2NzE0ZC02OTFiLTQxMDgtYTA2NC05NjRmMDQ1M2RmZGQiLCJ1bmlxdWVfbmFtZSI6ImFrYXNoLm11a2hlcmplZUBuem1pbmRzLmNvbSIsIm5hbWVpZCI6ImFrYXNoLm11a2hlcmplZUBuem1pbmRzLmNvbSIsImVtYWlsIjoiYWthc2gubXVraGVyamVlQG56bWluZHMuY29tIiwiYXV0aF90aW1lIjoiMDkvMDQvMjAyNSAwODoxMzo0MCIsImRiX25hbWUiOiJ3YXRpX2FwcF90cmlhbCIsImh0dHA6Ly9zY2hlbWFzLm1pY3Jvc29mdC5jb20vd3MvMjAwOC8wNi9pZGVudGl0eS9jbGFpbXMvcm9sZSI6IlRSSUFMIiwiZXhwIjoxNzU3NjM1MjAwLCJpc3MiOiJDbGFyZV9BSSIsImF1ZCI6IkNsYXJlX0FJIn0.A7DIELysmv6k0XRi7JNxRms2--dzW7ijiPa8WclE6Vc"
# }

# response = requests.post(url, headers=headers)

# print(response.status_code)
# print(response.text)
