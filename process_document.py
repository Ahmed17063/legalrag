import easyocr
from PIL import Image
from exchange import custom_prompt
from langchain.chat_models import ChatOpenAI  # or the one you're using
from typing import List
from utils import *


# Load EasyOCR model
ocr = easyocr.Reader(['en', 'fr'])

# You can pass your own chat instance and history
def generate_summary_from_image(image_path: str, chat, history: List, collection_name: str):
    # 1. Only use OCR
    result = ocr.readtext(image_path)
    lines = [res[1] for res in result]
    
    print("image content : " + str(lines))
    # 2. Enhanced prompt for LLM
    combined_query = f"""
    Extracted Text from Image:
    {' '.join(lines)}

    Please analyze this text and:
    1. Identify key concepts
    2. Highlight warnings/important notices
    3. Generate a concise summary
    """

    return custom_prompt(combined_query, chat, history, collection_name)


