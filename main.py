from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_groq import ChatGroq


load_dotenv()  # Load environment variables from .env file

llm_groq = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
    timeout=10,
)
response_groq = llm_groq.invoke("Hello, how are you?")
print(response_groq)