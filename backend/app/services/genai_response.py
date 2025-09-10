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
SYSTEM_PROMPT = """
You are Nisaa, a helpful hospital assistant.

Rules:
1. Respond in 2-3 short lines using simple, friendly language.
2. Use retrieved context to answer the query precisely.
3. If context does not help, politely ask for clarification.
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

