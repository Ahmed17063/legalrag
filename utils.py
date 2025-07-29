import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import os
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient, models

from langchain_community.document_loaders import DataFrameLoader
from langchain_openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI



load_dotenv()
qdrant_url = "http://localhost:6333" #os.getenv("QDRANT_URL") #"http://localhost:6333"
qdrant_api_key = os.getenv("QDRANT_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")
from db.db import *
params = get_parameters()

client_qdrant = QdrantClient(url=qdrant_url, api_key=qdrant_api_key) # native Qdrant client 

# Token limits for GPT model
Qdrant_limit = int(params.get("NB_LIMIT_QDRANT_RETRIEVES", 20))
MAX_TOKENS = int(params.get("OPENAI_MAX_TOKENS", 18000))
MINIMUM_CHUNK_SIZE = int(params.get("MINIMUM_CHUNK_SIZE", 300))
MAXIMUM_CHUNK_SIZE = int(params.get("MAXIMUM_CHUNK_SIZE", 600))
MAX_TRAINING_QUESTIONS = int(params.get("MAX_TRAINING_QUESTIONS", 5))
MODEL_NAME = "gpt-4o-mini"

# Initialize OpenAI embeddings (for text-to-vector transformation)
openai_embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=openai_api_key
)

chat_openai = ChatOpenAI(
    # model='gpt-4o'
     model='ft:gpt-4o-mini-2024-07-18:personal::Arl3TFMr'
    )

def crawl_website(base_url, visited=None, depth=2):
    print("crawling web site : " + base_url)
    if visited is None:
        visited = set()
    if depth == 0 or base_url in visited:
        return ""

    visited.add(base_url)
    try:
        response = requests.get(base_url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
    except requests.RequestException as e:
        print(f"Failed to fetch {base_url}: {e}")
        return ""

    # Extract main content from the page
    page_text = soup.get_text(separator="\n", strip=True)
    content = f"Content from {base_url}:\n{page_text}\n\n"
    return content

# Function to process the content in 500-character chunks
def process_in_chunks(content, chunk_size=500):
    for i in range(0, len(content), chunk_size):
        chunk = content[i:i + chunk_size]
        yield chunk  # Yield each chunk as a generator




    