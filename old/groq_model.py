from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage
)
import os
load_dotenv('./.env')

groq_api_key = os.getenv("GROQ_API_KEY")
chat = ChatGroq(temperature=0, model_name="mixtral-8x7b-32768")