# # from bs4 import BeautifulSoup
# # import requests

# # # url = 'https://en.wikipedia.org/wiki/List_of_largest_companies_in_the_United_States_by_revenue'

# # url= "https://webscraper.io/"

# # page = requests.get(url)

# # soup = BeautifulSoup(page.text, 'html')
# # print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>",soup)
# # # exit(0)   
# # soup.find('table')


# # soup.find_all('table')[1]

# # soup.find('table', class_ = 'wikitable sortable')


# # table = soup.find_all('table')[1]
# # print(table)

# # world_titles = table.find_all('th')
# # world_table_titles = [title.text.strip() for title in world_titles]

# # print(world_table_titles)

# # import pandas as pd
# # df = pd.DataFrame(columns = world_table_titles)

# # print(">>>>>>>>>>>>>>>df>>>>>>>>>>>>>>>>>>>",df)

# # column_data = table.find_all('tr')
# # for row in column_data[1:]:
# #     row_data = row.find_all('td')
# #     individual_row_data = [data.text.strip() for data in row_data]
    
# #     length = len(df)
# #     df.loc[length] = individual_row_data
# # print("......df..........",df)




# import requests
# # https://app-server.wati.io/api/v1/getMessages/919669092627
# your_api_token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJqdGkiOiJhNGE2NzE0ZC02OTFiLTQxMDgtYTA2NC05NjRmMDQ1M2RmZGQiLCJ1bmlxdWVfbmFtZSI6ImFrYXNoLm11a2hlcmplZUBuem1pbmRzLmNvbSIsIm5hbWVpZCI6ImFrYXNoLm11a2hlcmplZUBuem1pbmRzLmNvbSIsImVtYWlsIjoiYWthc2gubXVraGVyamVlQG56bWluZHMuY29tIiwiYXV0aF90aW1lIjoiMDkvMDQvMjAyNSAwODoxMzo0MCIsImRiX25hbWUiOiJ3YXRpX2FwcF90cmlhbCIsImh0dHA6Ly9zY2hlbWFzLm1pY3Jvc29mdC5jb20vd3MvMjAwOC8wNi9pZGVudGl0eS9jbGFpbXMvcm9sZSI6IlRSSUFMIiwiZXhwIjoxNzU3NjM1MjAwLCJpc3MiOiJDbGFyZV9BSSIsImF1ZCI6IkNsYXJlX0FJIn0.A7DIELysmv6k0XRi7JNxRms2--dzW7ijiPa8WclE6Vc"


# url = "https://app-server.wati.io/api/v1/getMessages/919669092627"
# headers = {
#     "Authorization": f"Bearer {your_api_token}"
# }

# # Send a GET request to the API
# # response = requests.get(url, headers=headers)
# # print(">>>>>>>>>>>>",response.status_code)
# # print(">>>>>>>>>>>>",response.json())
# # # Check if the request was successful
# # if response.status_code == 200:
# #     # Parse the JSON response
# #     resp_json = response.json()

# #     # Ensure 'data' exists and is a list
# #     messages = resp_json.get('data', [])
# #     if isinstance(messages, list) and messages:
# #         # Sort messages by 'timestamp' in descending order
# #         messages_sorted = sorted(messages, key=lambda x: x['timestamp'], reverse=True)

# #         # Retrieve the most recent message
# #         most_recent_message = messages_sorted[0]
# #         print("Most recent message:", most_recent_message)
# #     else:
# #         print("No messages found.")
# # else:
# #     print(f"Failed to retrieve messages. Status code: {response.status_code}")

# import requests
# # Assuming your get_embedding function is already defined above this code
# # from your_genai_module import get_embedding

# response = requests.get(url, headers=headers)
# # print(">>>>>>>>>>>>", response.status_code)
# # print(">>>>>>>>>>>>", response.json())

# # Check if the request was successful
# if response.status_code == 200:
#     # Parse the JSON response
#     data = response.json()
#     # Filter only user messages
#     user_messages = [
#         msg for msg in data['messages']['items'] 
#         if msg.get('type') == 'text' and msg.get('owner') is False
#     ]

#     if user_messages:
#         # Sort by timestamp to get the latest message
#         last_message = max(user_messages, key=lambda x: int(x['timestamp']))
#         print("Last user message:", last_message['text'])
#     else:
#         print("No user messages found.")