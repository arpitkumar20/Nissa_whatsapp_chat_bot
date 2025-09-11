# import asyncio
# from langchain_unstructured import UnstructuredLoader

# # page_url = "https://python.langchain.com/docs/how_to/chatbots_memory/"

# loader = UnstructuredLoader(web_url=page_url)

# async def main():
#     docs = []
#     async for doc in loader.alazy_load():
#         docs.append(doc)

#     # Show the first 500 characters of the first document
#     if docs:
#         # print("\n[INFO] Extracted Text Snippet:\n")
#         # print(docs[0].page_content[:500])
#         for doc in docs[:5]:
#             print(doc.page_content)
#             print(f"{doc.metadata['category']}: {doc.page_content}")

# # Run the async function
# asyncio.run(main())


# import bs4
# from langchain_community.document_loaders import WebBaseLoader
# import os

# loader = WebBaseLoader(web_path="https://www.nzminds.com/")
# # loader = WebBaseLoader(web_path="https://www.apollohospitals.com/hospitals/apollo-hospitals-kolkata")
# docs = loader.load()
# # print(docs)


# from langchain.text_splitter import RecursiveCharacterTextSplitter

# # Load example document
# # with open("state_of_the_union.txt") as f:
# #     state_of_the_union = f.read()

# text_splitter = RecursiveCharacterTextSplitter(
#     # Set a really small chunk size, just to show.
#     chunk_size=1000,
#     chunk_overlap=200,
#     # length_function=len,
#     # is_separator_regex=False,
# )
# texts = text_splitter.split_documents(docs)
# print(texts[0])
# print(texts[1])



# import requests
# from bs4 import BeautifulSoup

# # Set up headers to simulate a real browser
# headers = {
#     "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
#                   "AppleWebKit/537.36 (KHTML, like Gecko) "
#                   "Chrome/117.0.0.0 Safari/537.36"
# }

# # page_url = "https://www.nzminds.com/"
# # page_url = "https://www.apollohospitals.com/hospitals/apollo-hospitals-kolkata"
# page_url = "https://www.apollohospitals.com/doctors?f%5B0%5D=cities%3A297"
# # Make a GET request with headers
# response = requests.get(page_url, headers=headers)

# # Check status code
# if response.status_code == 200:
#     # Parse the HTML content
#     soup = BeautifulSoup(response.content, "html.parser")

#     # Extract desired part (example: the whole body content)
#     content = soup.get_text(separator="\n", strip=True)
    
#     print(content)
# else:
#     print(f"Failed to fetch page. Status code: {response.status_code}")

# element = soup.find(class_="theme-doc-markdown markdown")
# if element:
#     print(element.get_text(separator="\n", strip=True))
# else:
#     print("Specified element not found in page.")







# import requests
# from bs4 import BeautifulSoup
# from urllib.parse import urljoin, urlparse
# import time

# # Set up headers to simulate a real browser
# headers = {
#     "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
#                   "AppleWebKit/537.36 (KHTML, like Gecko) "
#                   "Chrome/117.0.0.0 Safari/537.36"
# }

# # base_url = "https://www.nzminds.com/"
# base_url = "https://www.apollohospitals.com/hospitals/apollo-hospitals-kolkata"

# parsed_base = urlparse(base_url)
# base_domain = f"{parsed_base.scheme}://{parsed_base.netloc}"

# visited_urls = set()
# to_visit_urls = [base_url]
# extracted_contents = []

# while to_visit_urls:
#     current_url = to_visit_urls.pop(0)
#     if current_url in visited_urls:
#         continue
#     visited_urls.add(current_url)
    
#     print(f"Fetching: {current_url}")
#     try:
#         response = requests.get(current_url, headers=headers, timeout=10)
#     except Exception as e:
#         print(f"Failed to fetch {current_url}: {e}")
#         continue

#     if response.status_code != 200:
#         print(f"Failed to fetch {current_url}: Status code {response.status_code}")
#         continue

#     soup = BeautifulSoup(response.content, "html.parser")

#     # Extract and store page text content
#     text_content = soup.get_text(separator="\n", strip=True)
#     extracted_contents.append({
#         "url": current_url,
#         "content": text_content
#     })

#     # Find all internal links
#     for link_tag in soup.find_all("a", href=True):
#         href = link_tag['href']
#         absolute_url = urljoin(base_domain, href)
#         parsed_url = urlparse(absolute_url)

#         # Only follow internal links
#         if parsed_url.netloc == parsed_base.netloc and absolute_url not in visited_urls:
#             if absolute_url not in to_visit_urls:
#                 to_visit_urls.append(absolute_url)

#     # Optional: polite crawling by adding delay
#     time.sleep(1)

# # Print summary
# print(f"\nTotal pages extracted: {len(extracted_contents)}")

# # Example: Print first page content
# if extracted_contents:
#     print("\nFirst page URL:", extracted_contents[0]['url'])
#     print("First page content preview:\n")
#     print(extracted_contents[0]['content'][:1000])  # Print first 1000 characters





# import requests
# from bs4 import BeautifulSoup
# from urllib.parse import urljoin, urlparse
# import time

# # Set up headers to simulate a real browser
# headers = {
#     "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
#                   "AppleWebKit/537.36 (KHTML, like Gecko) "
#                   "Chrome/117.0.0.0 Safari/537.36"
# }

# base_url = "https://www.nzminds.com/"
# parsed_base = urlparse(base_url)
# base_domain = f"{parsed_base.scheme}://{parsed_base.netloc}"

# visited_urls = set()
# to_visit_urls = [base_url]
# extracted_contents = []

# while to_visit_urls:
#     current_url = to_visit_urls.pop(0)
#     if current_url in visited_urls:
#         continue
#     visited_urls.add(current_url)
    
#     print(f"Fetching: {current_url}")
#     try:
#         response = requests.get(current_url, headers=headers, timeout=10)
#     except Exception as e:
#         print(f"Failed to fetch {current_url}: {e}")
#         continue

#     if response.status_code != 200:
#         print(f"Failed to fetch {current_url}: Status code {response.status_code}")
#         continue

#     soup = BeautifulSoup(response.content, "html.parser")

#     # Extract full text content
#     text_content = soup.get_text(separator="\n", strip=True)
#     extracted_contents.append({
#         "url": current_url,
#         "content": text_content
#     })

#     # Find all internal links
#     for link_tag in soup.find_all("a", href=True):
#         href = link_tag['href']
#         absolute_url = urljoin(base_domain, href)
#         parsed_url = urlparse(absolute_url)

#         # Only follow internal links from same domain
#         if parsed_url.netloc == parsed_base.netloc and absolute_url not in visited_urls:
#             if absolute_url not in to_visit_urls:
#                 to_visit_urls.append(absolute_url)

#     # Be polite
#     time.sleep(1)

# # Save data to text file
# output_file = "extracted_website_content.txt"
# with open(output_file, "w", encoding="utf-8") as f:
#     for page in extracted_contents:
#         f.write(f"URL: {page['url']}\n")
#         f.write("=" * 80 + "\n")
#         f.write(page['content'] + "\n\n\n")

# # Print summary
# print(f"\n✅ Total pages fetched and saved: {len(extracted_contents)}")
# print(f"✔️ Extracted data saved in file: {output_file}")





import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import time

# Set up headers to simulate a real browser
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/117.0.0.0 Safari/537.36"
}

# base_url = "https://www.nzminds.com/"
# base_url = "https://www.apollohospitals.com/hospitals/apollo-hospitals-kolkata"
# base_url= "https://charnockhospital.com/"
# base_url = "https://www.swagathgroups.com/hotel-swagath-kolkata/"
base_url = "https://cosmoshospital.org.in/"
parsed_base = urlparse(base_url)
base_domain = f"{parsed_base.scheme}://{parsed_base.netloc}"

visited_urls = set()
to_visit_urls = [base_url]
extracted_contents = []

page_counter = 1

while to_visit_urls:
    current_url = to_visit_urls.pop(0)
    if current_url in visited_urls:
        continue
    visited_urls.add(current_url)
    
    print(f"Fetching ({page_counter}): {current_url}")
    try:
        response = requests.get(current_url, headers=headers, timeout=10)
    except Exception as e:
        print(f"Failed to fetch {current_url}: {e}")
        continue

    if response.status_code != 200:
        print(f"Failed to fetch {current_url}: Status code {response.status_code}")
        continue

    soup = BeautifulSoup(response.content, "html.parser")

    # Extract full text content
    text_content = soup.get_text(separator="\n", strip=True)
    extracted_contents.append({
        "page_number": page_counter,
        "url": current_url,
        "content": text_content
    })

    # Find all internal links
    for link_tag in soup.find_all("a", href=True):
        href = link_tag['href']
        absolute_url = urljoin(base_domain, href)
        parsed_url = urlparse(absolute_url)

        # Only follow internal links from same domain
        if parsed_url.netloc == parsed_base.netloc and absolute_url not in visited_urls:
            if absolute_url not in to_visit_urls:
                to_visit_urls.append(absolute_url)

    page_counter += 1
    # Be polite
    time.sleep(1)

# Save structured data to text file
output_file = "extracted_hospital_website.txt"
# hotel_file = "hotel_details.txt"
with open(output_file, "w", encoding="utf-8") as f:
    for page in extracted_contents:
        f.write("-" * 50 + "\n")
        f.write(f"PAGE {page['page_number']}\n")
        f.write(f"URL: {page['url']}\n")
        f.write("-" * 50 + "\n\n")
        f.write(page['content'] + "\n\n\n")

    f.write(f"-----------------------------\n")
    f.write(f"TOTAL PAGES FETCHED: {len(extracted_contents)}\n")
    f.write(f"-----------------------------\n")

# Print summary
print(f"\n✅ Extraction complete!")
print(f"✔️ Total pages fetched: {len(extracted_contents)}")
print(f"✔️ Structured data saved in file: {output_file}")
