# import os
# import json
# from dotenv import load_dotenv
# # import genai
# # from genai.prompts import ChatPromptTemplate, MessagesPlaceholder
# # from genai.models import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# # from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
# # from genai.models import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
# import google.generativeai as genai
# from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# # Load environment variables
# load_dotenv()

# # from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

# # # Initialize the embeddings model
# # embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

# # # Initialize the chat model
# # llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro")

# GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
# GEMINI_MODEL = os.getenv('GEMINI_MODEL')
# EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL')
# LLM_TEMPERATURE = float(os.getenv('LLM_TEMPERATURE'))  # Default if not provided


# # 1️⃣ Configure GenAI
# genai.configure(api_key=GOOGLE_API_KEY)


# # 2️⃣ Initialize Models
# llm = ChatGoogleGenerativeAI(
#     model=GEMINI_MODEL,
#     temperature=LLM_TEMPERATURE,
#     google_api_key=GOOGLE_API_KEY
# )

# embeddings = GoogleGenerativeAIEmbeddings(
#     model=EMBEDDING_MODEL,
#     google_api_key=GOOGLE_API_KEY
# )

# lead_extraction_llm = ChatGoogleGenerativeAI(
#     model=GEMINI_MODEL,
#     temperature=0.1,
#     google_api_key=GOOGLE_API_KEY
# )


# # 3️⃣ Define Prompts

# CONTEXT_SYSTEM_PROMPT = """
# Given a chat history and the latest user question which might reference context in the chat history, 
# formulate a standalone question which can be understood without the chat history. 
# Do NOT answer the question, just reformulate it if needed and otherwise return it as is.
# """

# QA_SYSTEM_PROMPT = """
# You are Nisaa – the smart virtual assistant for a hospital chatbot.

# Your job is to:
# 1. Answer user questions strictly based on the provided context.
# 2. Provide clear, friendly responses in 2-3 lines.
# 3. Differentiate between general service inquiries and doctor consultations.
# 4. Follow the structured flow for booking services and doctor appointments.

# Important rules:
# - Ask for personal details (name, email, phone) in separate messages after context is addressed.
# - If unclear, ask the user to clarify.
# - Do not mix general service meetings with doctor appointments.

# Context: {context}
# Chat History: {chat_history}
# Question: {input}

# Answer strictly based on context in a short, helpful way.
# """

# LEAD_EXTRACTION_PROMPT = """
# Extract the following information from the conversation if available:
# - name
# - email_id  
# - contact_number
# - location
# - service_interest
# - appointment_date
# - appointment_time
# - doctor_name

# Return ONLY a valid JSON object with these fields with NO additional text before or after.  
# If information isn't found, leave the field empty.

# Do not include any explanatory text, notes, or code blocks. Return ONLY the raw JSON.

# Conversation: {conversation}
# """

# contextualize_q_prompt = ChatPromptTemplate.from_messages([
#     ("system", CONTEXT_SYSTEM_PROMPT),
#     MessagesPlaceholder("chat_history"),
#     ("human", "{input}"),
# ])

# qa_prompt = ChatPromptTemplate.from_messages([
#     ("system", QA_SYSTEM_PROMPT),
#     MessagesPlaceholder("chat_history"),
#     ("human", "{input}"),
# ])

# lead_extraction_prompt = ChatPromptTemplate.from_messages([
#     ("system", LEAD_EXTRACTION_PROMPT),
#     MessagesPlaceholder("chat_history"),
# ])


# # 4️⃣ Helper Functions

# def format_pinecone_context(pinecone_response):
#     context_parts = []
#     for entry in pinecone_response:
#         if entry.get("type") == "doctor":
#             context_parts.append(
#                 f"Doctor Name: {entry.get('name', '')}, Phone: {entry.get('phone', '')}, "
#                 f"Profile: {entry.get('profile_url', '')}, Details: {entry.get('text', '')}"
#             )
#         else:
#             context_parts.append(
#                 f"Page Title: {entry.get('title', '')}, URL: {entry.get('source_url', '')}, "
#                 f"Text Snippet: {entry.get('text', '')[:300]}..."
#             )
#     return "\n\n".join(context_parts)


# def get_contextual_question(chat_history, user_input):
#     prompt = contextualize_q_prompt.format(chat_history=chat_history, input=user_input)
#     return llm.predict(prompt).strip()


# def get_answer(chat_history, user_input, context):
#     prompt = qa_prompt.format(context=context, chat_history=chat_history, input=user_input)
#     return llm.predict(prompt).strip()


# def extract_lead_data(chat_history):
#     full_conversation = "\n".join(chat_history)
#     prompt = lead_extraction_prompt.format(conversation=full_conversation)
#     response = lead_extraction_llm.predict(prompt)
#     try:
#         return json.loads(response)
#     except json.JSONDecodeError:
#         return {}


# # 5️⃣ Main Controller Flow

# def handle_user_query(chat_history, user_input, pinecone_response):
#     # Step 1: Build context from Pinecone DB
#     context = format_pinecone_context(pinecone_response)

#     # Step 2: Reformulate the user's question
#     standalone_question = get_contextual_question(chat_history, user_input)

#     # Step 3: Get chatbot's answer based on the context
#     answer = get_answer(chat_history, standalone_question, context)

#     # Step 4: Extract lead data
#     lead_data = extract_lead_data(chat_history + [user_input])

#     return {
#         "standalone_question": standalone_question,
#         "answer": answer,
#         "lead_data": lead_data
#     }

# # 6️⃣ Example Usage

# if __name__ == "__main__":
#     chat_history = [
#         "Hi, I'm interested in booking a doctor appointment."
#     ]
#     user_input = "Can you show me a list of cardiologists available?"
    
#     # Example Pinecone response (shortened for demonstration)
#     pinecone_response = [
#             {
#             "id": "page_-2825318644890053292",
#             "score": 0.672163367,
#             "site_id": "apollo",
#             "site_name": "Apollo Hospitals",
#             "source_url": "https://bansalhospital.com/",
#             "text": "cular surgeons… Select the concern area, we will guide you. Eye Ear Teeth Throat Chest Liver Arm Pancreas Kidneys Lower Back Thigh Foot Brain Nose Heart Lungs Elbow Spine Stomach Hip Knee Ankle Eye Ear Teeth Throat Breast Liver Arm Pancreas Kidneys Lower Back Reproductive systems Thigh Foot Brain Nose Heart Lungs Elbow Spine Stomach Hip Knee Ankle Meet our team Chief Executive Officer (CEO) -- Administration Specialization and… Consultant -- Internal Medicine & Rheumatology Specialization and… Consultant - Cardiology Specialization and… Consultant -- Neurology Specialization and… Consultant -- Internal Medicine & Rheumatology Specialization and… Consultant -- Neuro Surgery Specialization and… Consultant -- Internal Medicine & Rheumatology Specialization and… Consultant -- Onco Surgery Specialization and… Consultant -- Cardio Thoracic and Vascular Surgery Specialization and… Consultant -- Gastroenterologist , Hepatologist & Liver Transplant Physician Specialization and… Consultant -- Ca",
#             "title": "ABOUTBANSAL HOSPITAL",
#             "type": "page"
#         },
#         {
#             "id": "page_-8539165227784746369",
#             "score": 0.654438853,
#             "site_id": "apollo",
#             "site_name": "Apollo Hospitals",
#             "source_url": "https://bansalhospital.com/doctors-search",
#             "text": " Consultant -- Nuclear Medicine Specialization and Expertise… Consultant -- Neuro Surgery Specialization and Expertise… Consultant - Anesthesiologist ( Cardiac & Neuro ) Specialization and Expertise… Consultant -- Intervention Pain Management Specialization and Expertise… HOD - Anaesthesia Specialization and Expertise… Consultant -- Dental Specialization and Expertise… Anaesthesia Specialization and Expertise… Consultant -- Radiation Oncology Specialization and Expertise… Consultant -- Dental Specialization and Expertise… Consultant -- Endocrinology Specialization and Expertise… Consultant -- Rheumatology & Immunology Specialization and Expertise… Consultant -- Ophthalmology Work as comprehensive… Consultant -- Gastro and Advanced laparoscopic Surgeon Specialization and Expertise… Consultant -- Radiation Oncology Specialization and Expertise… Consultant -- Orthopedics Specialization and Expertise… Consultant -- Obstetrics and Gynaecology Specialization and Expertise… Consultant -- Card",
#             "title": "Bansal Hospital Bhopal Doctor List",
#             "type": "page"
#         }
#     ]

#     result = handle_user_query(chat_history, user_input, pinecone_response)

#     print("Standalone Question:")
#     print(result["standalone_question"])
#     print("\nChatbot Answer:")
#     print(result["answer"])
#     print("\nExtracted Lead Data:")
#     print(result["lead_data"])






# import os
# import json
# import logging
# from dotenv import load_dotenv
# import google.generativeai as genai
# from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# from langchain_core.messages import HumanMessage

# # Setup logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # Load environment variables
# load_dotenv()

# GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
# GEMINI_MODEL = os.getenv('GEMINI_MODEL')
# EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL')
# LLM_TEMPERATURE = float(os.getenv('LLM_TEMPERATURE', 0.7))

# logger.info("Configuring GenAI with provided API key.")
# genai.configure(api_key=GOOGLE_API_KEY)

# logger.info("Initializing models...")
# llm = ChatGoogleGenerativeAI(
#     model=GEMINI_MODEL,
#     temperature=LLM_TEMPERATURE,
#     google_api_key=GOOGLE_API_KEY
# )

# embeddings = GoogleGenerativeAIEmbeddings(
#     model=EMBEDDING_MODEL,
#     google_api_key=GOOGLE_API_KEY
# )

# lead_extraction_llm = ChatGoogleGenerativeAI(
#     model=GEMINI_MODEL,
#     temperature=0.1,
#     google_api_key=GOOGLE_API_KEY
# )

# # Prompts
# CONTEXT_SYSTEM_PROMPT = """
# Given the context from hospital information database,
# answer the user question in a helpful, friendly way.
# """

# QA_SYSTEM_PROMPT = """
# You are Nisaa – the hospital assistant chatbot.

# Answer the question based solely on the following context in a short 2-3 lines.
# Do NOT add personal info or ask for input.

# Context: {context}

# Answer:
# """

# contextualize_q_prompt = ChatPromptTemplate.from_messages([
#     ("system", CONTEXT_SYSTEM_PROMPT),
#     MessagesPlaceholder(variable_name="messages"),
# ])

# qa_prompt = ChatPromptTemplate.from_messages([
#     ("system", QA_SYSTEM_PROMPT),
#     MessagesPlaceholder(variable_name="messages"),
# ])

# def format_pinecone_context(pinecone_response):
#     logger.info("Formatting context from Pinecone DB response...")
#     context_parts = []
#     for entry in pinecone_response:
#         if entry.get("type") == "doctor":
#             context_parts.append(
#                 f"Doctor Name: {entry.get('name', '')}, Phone: {entry.get('phone', '')}, "
#                 f"Profile: {entry.get('profile_url', '')}, Details: {entry.get('text', '')}"
#             )
#         else:
#             context_parts.append(
#                 f"Page Title: {entry.get('title', '')}, URL: {entry.get('source_url', '')}, "
#                 f"Text Snippet: {entry.get('text', '')[:300]}..."
#             )
#     context = "\n\n".join(context_parts)
#     logger.info("Context formatted successfully.")
#     return context

# def get_chatbot_answer(pinecone_response):
#     context = format_pinecone_context(pinecone_response)
#     logger.info("Generating final answer from context...")

#     messages = [HumanMessage(content=context)]
#     prompt = qa_prompt.format(messages=messages, context=context)

#     response = llm.predict(prompt).strip()
#     logger.info("Answer generated successfully.")

#     return response

# # Example Main Controller (for direct use)
# def main(pinecone_response):
#     logger.info("Starting chatbot processing...")
#     answer = get_chatbot_answer(pinecone_response)
#     logger.info("Chatbot processing complete.")

#     return {
#         "answer": answer
#     }


# if __name__ == "__main__":
#     sample_pinecone_response = [
#             {
#             "id": "page_-2825318644890053292",
#             "score": 0.672163367,
#             "site_id": "apollo",
#             "site_name": "Apollo Hospitals",
#             "source_url": "https://bansalhospital.com/",
#             "text": "cular surgeons… Select the concern area, we will guide you. Eye Ear Teeth Throat Chest Liver Arm Pancreas Kidneys Lower Back Thigh Foot Brain Nose Heart Lungs Elbow Spine Stomach Hip Knee Ankle Eye Ear Teeth Throat Breast Liver Arm Pancreas Kidneys Lower Back Reproductive systems Thigh Foot Brain Nose Heart Lungs Elbow Spine Stomach Hip Knee Ankle Meet our team Chief Executive Officer (CEO) -- Administration Specialization and… Consultant -- Internal Medicine & Rheumatology Specialization and… Consultant - Cardiology Specialization and… Consultant -- Neurology Specialization and… Consultant -- Internal Medicine & Rheumatology Specialization and… Consultant -- Neuro Surgery Specialization and… Consultant -- Internal Medicine & Rheumatology Specialization and… Consultant -- Onco Surgery Specialization and… Consultant -- Cardio Thoracic and Vascular Surgery Specialization and… Consultant -- Gastroenterologist , Hepatologist & Liver Transplant Physician Specialization and… Consultant -- Ca",
#             "title": "ABOUTBANSAL HOSPITAL",
#             "type": "page"
#         },
#         {
#             "id": "page_-8539165227784746369",
#             "score": 0.654438853,
#             "site_id": "apollo",
#             "site_name": "Apollo Hospitals",
#             "source_url": "https://bansalhospital.com/doctors-search",
#             "text": " Consultant -- Nuclear Medicine Specialization and Expertise… Consultant -- Neuro Surgery Specialization and Expertise… Consultant - Anesthesiologist ( Cardiac & Neuro ) Specialization and Expertise… Consultant -- Intervention Pain Management Specialization and Expertise… HOD - Anaesthesia Specialization and Expertise… Consultant -- Dental Specialization and Expertise… Anaesthesia Specialization and Expertise… Consultant -- Radiation Oncology Specialization and Expertise… Consultant -- Dental Specialization and Expertise… Consultant -- Endocrinology Specialization and Expertise… Consultant -- Rheumatology & Immunology Specialization and Expertise… Consultant -- Ophthalmology Work as comprehensive… Consultant -- Gastro and Advanced laparoscopic Surgeon Specialization and Expertise… Consultant -- Radiation Oncology Specialization and Expertise… Consultant -- Orthopedics Specialization and Expertise… Consultant -- Obstetrics and Gynaecology Specialization and Expertise… Consultant -- Card",
#             "title": "Bansal Hospital Bhopal Doctor List",
#             "type": "page"
#         }
#     ]


#     result = main(sample_pinecone_response)
#     print("Generated Answer:")
#     print(result["answer"])


import os
import json
import logging
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import HumanMessage
import google.generativeai as genai

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
GEMINI_MODEL = os.getenv('GEMINI_MODEL')
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL')
LLM_TEMPERATURE = float(os.getenv('LLM_TEMPERATURE', 0.7))

# Configure GenAI
genai.configure(api_key=GOOGLE_API_KEY)

# Initialize Models
llm = ChatGoogleGenerativeAI(
    model=GEMINI_MODEL,
    temperature=LLM_TEMPERATURE,
    google_api_key=GOOGLE_API_KEY
)

embeddings = GoogleGenerativeAIEmbeddings(
    model=EMBEDDING_MODEL,
    google_api_key=GOOGLE_API_KEY
)

lead_extraction_llm = ChatGoogleGenerativeAI(
    model=GEMINI_MODEL,
    temperature=0.1,
    google_api_key=GOOGLE_API_KEY
)

# # Define Prompts
# CONTEXT_SYSTEM_PROMPT = """
# Given a chat history and the latest user question which might reference context in the chat history, 
# formulate a standalone question which can be understood without the chat history. 
# Do NOT answer the question, just reformulate it if needed and otherwise return it as is.
# """

# QA_SYSTEM_PROMPT = """
# You are Nisaa – the smart virtual assistant for a hospital chatbot.

# Your job is to:
# 1. Answer user questions strictly based on the provided context.
# 2. Provide clear, friendly responses in 2-3 lines.
# 3. Differentiate between general service inquiries and doctor consultations.
# 4. Follow the structured flow for booking services and doctor appointments.

# Important rules:
# - Ask for personal details (name, email, phone) in separate messages after context is addressed.
# - If unclear, ask the user to clarify.
# - Do not mix general service meetings with doctor appointments.

# Context: {context}

# Answer strictly based on context in a short, helpful way.
# """

# LEAD_EXTRACTION_PROMPT = """
# Extract the following information from the conversation if available:
# - name
# - email_id  
# - contact_number
# - location
# - service_interest
# - appointment_date
# - appointment_time
# - doctor_name

# Return ONLY a valid JSON object with these fields with NO additional text before or after.  
# If information isn't found, leave the field empty.

# Do not include any explanatory text, notes, or code blocks. Return ONLY the raw JSON.

# Conversation: {conversation}
# """

# qa_prompt = ChatPromptTemplate.from_messages([
#     ("system", QA_SYSTEM_PROMPT),
#     ("human", "{context}"),
# ])

# lead_extraction_prompt = ChatPromptTemplate.from_messages([
#     ("system", LEAD_EXTRACTION_PROMPT),
#     ("human", "{conversation}"),
# ])

# # Helper Functions
# def format_pinecone_context(pinecone_response):
#     logger.info("Formatting Pinecone context.")
#     context_parts = []
#     for entry in pinecone_response:
#         if entry.get("type") == "doctor":
#             context_parts.append(
#                 f"Doctor Name: {entry.get('name', '')}, Phone: {entry.get('phone', '')}, "
#                 f"Profile: {entry.get('profile_url', '')}, Details: {entry.get('text', '')}"
#             )
#         else:
#             context_parts.append(
#                 f"Page Title: {entry.get('title', '')}, URL: {entry.get('source_url', '')}, "
#                 f"Text Snippet: {entry.get('text', '')[:300]}..."
#             )
#     return "\n\n".join(context_parts)

# def get_answer(context):
#     logger.info("Generating answer using context.")
#     prompt = qa_prompt.format(context=context)
#     return llm.predict(prompt).strip()

# def extract_lead_data(context):
#     logger.info("Extracting lead data from context.")
#     prompt = lead_extraction_prompt.format(conversation=context)
#     response = lead_extraction_llm.predict(prompt)
#     try:
#         return json.loads(response)
#     except json.JSONDecodeError:
#         logger.warning("Lead extraction returned invalid JSON.")
#         return {}

# # Main Controller Flow
# def handle_user_query(pinecone_response):
#     logger.info("Handling user query based solely on Pinecone DB response.")

#     # Step 1: Build context from Pinecone DB
#     context = format_pinecone_context(pinecone_response)

#     # Step 2: Generate chatbot's answer
#     answer = get_answer(context)

#     # Step 3: Extract lead data
#     lead_data = extract_lead_data(context)

#     return {
#         "answer": answer,
#         "lead_data": lead_data
#     }





# === Prompts & LLMs (keep your existing LLM/embedding init) ===

CONTEXT_SYSTEM_PROMPT = """
Given a chat history and the latest user question which might reference context in the chat history,
formulate a single standalone question which can be understood without chat history.
Do NOT answer — only return the reformulated question (or the same question if no change needed).
Return plain text only.
"""

QA_SYSTEM_PROMPT = """
You are Nisaa — the smart virtual assistant for a hospital chatbot.

Rules (must follow):
1) Answer strictly using ONLY the provided Context. If the answer is not present in Context, say you don't have that info and ask for clarification.
2) Provide concise, friendly responses of 2-3 lines (max). Use plain language.
3) When you reference specific facts (doctor name, hospital, phone, address), include the source URL in parentheses after the fact if available.
4) Differentiate clearly between general-service queries (e.g., "What are visiting hours?") and doctor consultations (e.g., "Book appointment with Dr X"). If the user asks for booking, outline the booking flow in 2 steps and then ask for personal info in a separate message.
5) Do NOT ask for personal contact details (name, phone, email) in the same message as clinical or context information. Request them in a follow-up message only.
6) If user intent is ambiguous, ask one clarifying question.

Context (do NOT invent anything): 
{context}

Respond in 2-3 lines and adhere to rules above.
"""

LEAD_EXTRACTION_PROMPT = """
Extract the following fields from the Conversation text or structured context. Return ONLY a valid JSON object and nothing else.

Fields to extract (leave empty string "" if not present):
- name
- email_id
- contact_number
- location
- service_interest
- appointment_date   (ISO format if possible: YYYY-MM-DD)
- appointment_time   (24h if possible: HH:MM)
- doctor_name
- source_url         (where the info was found)

You may pull data from textual snippets or embedded JSON-LD (if present). Do NOT add commentary or extra keys.

Conversation / Context: {conversation}
"""

# Use ChatPromptTemplate as you had, updated messages:
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", QA_SYSTEM_PROMPT),
    ("human", "{context}"),
])

lead_extraction_prompt = ChatPromptTemplate.from_messages([
    ("system", LEAD_EXTRACTION_PROMPT),
    ("human", "{conversation}"),
])

# === Improved context formatting function ===

def safe_get_jsonld(entry):
    """
    Try to parse 'jsonld' if present and return a dict or None.
    """
    import json
    jsonld = entry.get("jsonld") or entry.get("json_ld") or entry.get("json_ld_str")
    if not jsonld:
        return None
    try:
        # jsonld may be stringified list/dict; try loads
        parsed = json.loads(jsonld)
        return parsed
    except Exception:
        return None

def extract_contact_from_jsonld(parsed_jsonld):
    """
    Look for telephone, address, name, url, medicalSpecialty inside parsed jsonld.
    Return a dict of discovered values.
    """
    found = {}
    try:
        # parsed_jsonld may be a list of graph items or a single dict
        candidates = parsed_jsonld if isinstance(parsed_jsonld, list) else [parsed_jsonld]
        for item in candidates:
            # flatten common fields
            if isinstance(item, dict):
                if not found.get("name") and item.get("name"):
                    found["name"] = item.get("name")
                if not found.get("telephone") and item.get("telephone"):
                    found["telephone"] = item.get("telephone")
                if not found.get("url") and item.get("url"):
                    found["url"] = item.get("url")
                if not found.get("address") and item.get("address"):
                    addr = item.get("address")
                    if isinstance(addr, dict):
                        addr_str = ", ".join([addr.get(k,"") for k in ("streetAddress","addressLocality","addressRegion","postalCode") if addr.get(k)])
                        found["address"] = addr_str
                # medicalSpecialty could be list/dict
                if not found.get("medicalSpecialty") and item.get("medicalSpecialty"):
                    ms = item.get("medicalSpecialty")
                    if isinstance(ms, list):
                        found["medicalSpecialty"] = ", ".join([m.get("name") if isinstance(m, dict) else str(m) for m in ms])
                    elif isinstance(ms, dict):
                        found["medicalSpecialty"] = ms.get("name")
    except Exception:
        pass
    return found

def format_pinecone_context(pinecone_response):
    """
    Builds a helpful context string from a list of Pinecone entries.
    Each entry can be a doctor, hospital, or generic page. We inspect keys:
      - type, namespace, title, source_url, text, telephone, address, jsonld
    We attempt to parse jsonld and pick contact/doctor/hospital info when present.
    """
    logger.info("Formatting Pinecone context.")
    context_parts = []
    for entry in pinecone_response:
        # common direct fields
        entry_type = entry.get("type") or entry.get("namespace") or "page"
        title = entry.get("title") or entry.get("name") or ""
        url = entry.get("source_url") or entry.get("url") or entry.get("profile_url") or ""
        text = entry.get("text") or entry.get("description") or ""
        telephone = entry.get("telephone") or entry.get("phone") or ""
        address = entry.get("address") or ""
        # try jsonld parsing for richer fields
        parsed_jsonld = safe_get_jsonld(entry)
        jsonld_info = extract_contact_from_jsonld(parsed_jsonld) if parsed_jsonld else {}

        # prefer jsonld values when available
        if jsonld_info.get("name"):
            title = title or jsonld_info.get("name")
        if jsonld_info.get("telephone"):
            telephone = telephone or jsonld_info.get("telephone")
        if jsonld_info.get("address"):
            address = address or jsonld_info.get("address")
        medical_specialty = jsonld_info.get("medicalSpecialty") or entry.get("medicalSpecialty")  # could be string or list

        # format based on detected type
        if entry_type.lower() in ("doctor", "physician", "practitioner"):
            # doctor-specific
            doctor_info = [
                f"Doctor Name: {title}" if title else "Doctor Name: ",
                f"Specialty: {medical_specialty}" if medical_specialty else "",
                f"Phone: {telephone}" if telephone else "",
                f"Profile URL: {url}" if url else "",
                f"Snippet: {text[:300]}..." if text else ""
            ]
            context_parts.append(" | ".join([p for p in doctor_info if p]))
        elif entry.get("namespace") == "hospital" or entry_type.lower() in ("hospital", "medicalorganization", "medical_organization"):
            # hospital-specific
            hosp_info = [
                f"Hospital: {title}" if title else "Hospital:",
                f"Address: {address}" if address else "",
                f"Phone: {telephone}" if telephone else "",
                f"Specialties: {medical_specialty}" if medical_specialty else "",
                f"URL: {url}" if url else "",
                f"Snippet: {text[:300]}..." if text else ""
            ]
            context_parts.append(" | ".join([p for p in hosp_info if p]))
        else:
            # generic page
            page_info = [
                f"Page Title: {title}" if title else "Page Title:",
                f"URL: {url}" if url else "",
                f"Snippet: {text[:300]}..." if text else ""
            ]
            context_parts.append(" | ".join([p for p in page_info if p]))

    # join with clear separators
    return "\n\n".join(context_parts)


# === Answering and lead extraction helpers ===

def get_answer(context):
    """
    Formats the prompt with the assembled context and queries the llm.
    Returned string is the assistant's short reply.
    """
    logger.info("Generating answer using context.")
    prompt = qa_prompt.format(context=context)
    # Use llm.predict or .generate depending on your SDK; retained .predict for compatibility
    return llm.predict(prompt).strip()

def extract_lead_data(context):
    """
    Uses the lead extraction LLM to produce a strict JSON.
    Attempts to parse the LLM output; returns a dict or empty dict on parse error.
    """
    logger.info("Extracting lead data from context.")
    prompt = lead_extraction_prompt.format(conversation=context)
    response = lead_extraction_llm.predict(prompt)
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        logger.warning("Lead extraction returned invalid JSON; attempting best-effort extraction.")
        # Best-effort: try to extract common fields with regex fallback (very permissive)
        fallback = {"name": "", "email_id": "", "contact_number": "", "location": "", "service_interest": "", "appointment_date": "", "appointment_time": "", "doctor_name": "", "source_url": ""}
        # simple regexes
        import re
        email = re.search(r"([\w\.-]+@[\w\.-]+\.\w+)", response)
        phone = re.search(r"(\+?\d[\d\s\-]{6,}\d)", response)
        date = re.search(r"(\d{4}-\d{2}-\d{2})", response)
        time = re.search(r"([01]?\d|2[0-3]):[0-5]\d", response)
        if email: fallback["email_id"] = email.group(1)
        if phone: fallback["contact_number"] = phone.group(1)
        if date: fallback["appointment_date"] = date.group(1)
        if time: fallback["appointment_time"] = time.group(1)
        # return fallback (may be partially empty)
        return fallback

# === Main controller ===

def handle_user_query(pinecone_response):
    """
    Controller that:
      1) builds context (from pinecone_response)
      2) obtains a short assistant answer
      3) extracts lead data as JSON
    """
    logger.info("Handling user query based solely on Pinecone DB response.")
    context = format_pinecone_context(pinecone_response)
    answer = get_answer(context)
    lead_data = extract_lead_data(context)
    return {
        "answer": answer,
        "lead_data": lead_data
    }



