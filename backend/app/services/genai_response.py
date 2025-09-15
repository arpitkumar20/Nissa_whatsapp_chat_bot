import os
import json
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# === Configuration ===
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
GEMINI_MODEL = os.getenv('GEMINI_MODEL')
LLM_TEMPERATURE = float(os.getenv('LLM_TEMPERATURE', 0.7))

# Configure GenAI
genai.configure(api_key=GOOGLE_API_KEY)

# Initialize GenAI model
llm = ChatGoogleGenerativeAI(
    model=GEMINI_MODEL,
    temperature=LLM_TEMPERATURE,
    google_api_key=GOOGLE_API_KEY
)

# System prompt to guide GenAI
# SYSTEM_PROMPT = """
# You are Nisaa, a helpful hospital assistant.

# Your role:
# - Use the provided context (from Pinecone DB) to answer user questions about hospital and doctors.
# - The context may include doctor names, specialties, timings, availability, hospital details, etc.

# Response rules:
# 1. Keep answers short (2–3 sentences) and friendly.
# 2. Always answer using only the retrieved context.
# 3. If the query is about a list (e.g., "list of cardiologists"), return a clean, simple list.
# 4. If context does not provide enough information, politely ask the user for clarification instead of guessing.
# 5. Do not invent or assume any details not present in the context.

# Example behaviors:
# - User: "Which doctors are available for cardiology?"  
#   Nisaa: "Here are the cardiologists: Dr. A Sharma (Mon–Fri, 10am–2pm), Dr. B Khan (Sat–Sun, 4pm–8pm)."

# - User: "Tell me about Dr. Meera."  
#   Nisaa: "Dr. Meera is a pediatric specialist available Mon–Sat, 9am–1pm. Do you want appointment details?"
# """

SYSTEM_PROMPT = """
You are chatbot, an intelligent assistant. Your role is to answer user queries strictly based on the provided context, ensuring responses are accurate, relevant, and fully supported by that context.

Instructions:
- Answer strictly using the provided context only.
- Ensure every answer matches the user query exactly and is fully supported by the context.
- Keep responses short, clear, and friendly.
- If the context does not provide enough information, politely ask the user for clarification instead of guessing.
- Never add, assume, or invent details not present in the context.
"""


# Define a simple prompt template
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "Relevant Context:\n{context}\n\nUser Question:\n{query}")
])

# Core function to handle the query
def handle_user_query(retrieved_context: str, query: str) -> str:
    # Ensure inputs are valid strings
    retrieved_context = retrieved_context or "No relevant context available."
    query = query or "No query provided."

    # Format the prompt
    formatted_prompt = qa_prompt.format(
        context=retrieved_context,
        query=query
    )

    # Generate response
    response = llm.predict(formatted_prompt).strip()

    return response


# === Example Usage ===
# pinecone_context = "Previous queries show available appointment slots on Friday from 10 AM to 3 PM."
# user_query = "Can I book an appointment on Friday afternoon?"

# response = handle_user_query(
#     retrieved_context=pinecone_context,
#     query=user_query
# )

# print("\n🌟 GenAI Response:\n", response)

