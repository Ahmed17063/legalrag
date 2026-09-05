from pydantic import BaseModel
from fastapi.responses import JSONResponse
from typing import List
from fastapi.middleware.cors import CORSMiddleware
# from docs_process.docx_processor import process_docx
from docs_process.docx_processor import process_docx
from evaluation.evaluation_ragas import Evaluator
from exchange import get_response, get_title_and_intention, get_conversation_title_and_intention
from twilio.rest import Client
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import PlainTextResponse
from typing import Literal
from langdetect import detect
from process_document import generate_summary_from_image
from utils import chat_openai
import numpy as np
from qdrant_local import store_use_cases_in_qdrant
app = FastAPI()

# Add CORS middleware to allow requests from any origin (use this for development)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins, you can specify specific domains if needed
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

# Define a Pydantic model for the incoming data
class Message(BaseModel):
    sender: Literal["user", "bot"]
    content: str

class UserMessage(BaseModel):
    session_id: str
    messages: List[Message]
    collection_name: str = "client"
    image_url: str | None = None


from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage
)

@app.post("/chat")
async def bot(user_message: UserMessage):
    if not user_message.messages:
        return JSONResponse(content={"response": "Message list is empty."}, status_code=400)

    # Extract latest user message
    user_query = next((m.content for m in reversed(user_message.messages) if m.sender == "user"), None)
    if not user_query:
        return JSONResponse(content={"response": "No valid user message."}, status_code=400)



    user_lang = detect(user_query)
    # Format LangChain-compatible history
    formatted_messages = [
        SystemMessage(content="You are a helpful assistant ONLY for the clients and agents of the Banque Mauritanienne pour le Commerce International (BMCI) and the clients and agents of Masrvi, which is a BMCI wallet. Their websites are bmci.mr and masrvi.mr, you can communicate in English, French and arabic. The client might send screenshots of BMCI or Masrvi applications you may answer that. YOU DO NOT ANSWER ANYHTING THAT IS NOT RELATED TO BMCI or Masrvi, NEVER. You can provide analysis or specific information based on screenshots and if the client asks about procedures to do something you can answer based on the informations you have. Your answer should be between 200 and 500 caracters. You can give BMCI and Masrvi apps donwloads links , web sites and anything related that will be given to you in the contexts. Use often BCMI abreviation instead on the whole name. Always respond like an employee of BMCI, so NEVER say something like their website, intead say our website. if the score if > 0.7 answer with confident and don't send the contact number. Answer in the {user_lang} language"),
    ]

    for msg in user_message.messages:
        if msg.sender == "user":
            formatted_messages.append(HumanMessage(content=msg.content))
        elif msg.sender == "bot":
            formatted_messages.append(AIMessage(content=msg.content))

    print(f"Collection name : {user_message.collection_name}")
    # Get context-aware response
    # response = custom_prompt(query=user_query, chat=chat_openai, history=formatted_messages, collection_name=user_message.collection_name)
    # print("user prompt : " + user_message.messages[-1].content)
    if user_message.image_url:
        print(f"Received image URL: {user_message.image_url}")
        try:
            response = generate_summary_from_image(
                image_path="../chatbot service/public/uploads/"+ user_message.image_url,
                chat=chat_openai,
                history=formatted_messages,
                collection_name=user_message.collection_name
            )
        except Exception as e:
            print(f"Error processing image: {e}")
            response = "There was an error analyzing the image. Please try again or send another image."
    else:
        response = get_response(
                    query=user_query,
                    chat=chat_openai,
                    history=formatted_messages,
                    collection_name=user_message.collection_name
                )
        
        # Step 2: Get title/intention ONLY for evaluation
        etiquette_labels = get_etiquettes()  # Fetch once globally if possible!
        title_intention = get_title_and_intention(
            query = formatted_messages,
            chat=chat_openai,
            etiquette_labels=etiquette_labels
        )
    # Append bot response
    updated_history = user_message.messages + [{"sender": "bot", "content": response}]

    return {
        "response": response["response"],
        "history": updated_history,
        "title": title_intention["title"],
        "category": title_intention["category"],
        "intention" : title_intention["intention"]
        }



import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging
from io import BytesIO
from langchain.schema import SystemMessage
from sentence_transformers import util

# Configure logging
logging.basicConfig(filename='rag_evaluation.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')



from fastapi.responses import JSONResponse
from langchain.schema import SystemMessage
from sentence_transformers import util
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
import pandas as pd
import time
import asyncio
import logging
import numpy as np
import os
import math

from ragas.evaluation import evaluate

from ragas import SingleTurnSample  # for single-turn QA
from ragas import evaluate
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall


@app.post("/evaluate")
async def evaluate_rag(
    file: UploadFile = File(...),  # Receive file directly
    collection_name: str = "clients"  # Optional: if you need collection name
):
    temp_path = f"/tmp/{file.filename}"
    with open(temp_path, "wb") as buffer:
        buffer.write(await file.read())
    if not os.path.exists(temp_path):
        return JSONResponse(content={"error": "Asset file not found"}, status_code=404)
    
    evaluator = Evaluator()
    report = evaluator.evaluate(
        file_path=temp_path,
        column_mapping={
            'qa_id': "ID",
            'question': 'Question Utilisateur',
            'ground_truth': 'Réponse Attendue',
            'intention': 'Intention',
            'category': 'Catégorie',
            'expected_contexts': 'Contextes Attendus (si RAG)',
        },
        max_rows=MAX_TRAINING_QUESTIONS
    )

    return JSONResponse(content=report, status_code=200)


@app.post("/chargeUseCases")
async def charge_use_cases(
    file: UploadFile = File(...),  # Receive file directly
    collection_name: str = "clients"  # Optional: if you need collection name
):
    temp_path = f"/tmp/{file.filename}"
    with open(temp_path, "wb") as buffer:
        buffer.write(await file.read())
    if not os.path.exists(temp_path):
        return JSONResponse(content={"error": "Asset file not found"}, status_code=404)
    
    report = store_use_cases_in_qdrant(file_path=temp_path)
    return JSONResponse(content=report, status_code=200)


@app.post("/chargeProductsAndServices")
async def chargeProductsAndServices(
    file: UploadFile = File(...),  # Receive file directly
    collection_name: str = "clients"  # Optional: if you need collection name
):
    temp_path = f"/tmp/{file.filename}"
    with open(temp_path, "wb") as buffer:
        buffer.write(await file.read())
    if not os.path.exists(temp_path):
        return JSONResponse(content={"error": "Asset file not found"}, status_code=404)
    
    report = process_docx(file_path=temp_path)
    print(f"Processed {len(report)} chunks from {file.filename}")
    print(report)
    return JSONResponse(content=report, status_code=200)




from qdrant_local import recreate_collection,store_data_in_collection, set_collection_indexing  # Import both LangChain wrapper and native Qdrant client
from utils import *
import time 
import pandas as pd

text =[
    "Conditions Générales : Aucune information apparaissant sur le présent site ne saurait être",
    "Voici les lient pour telecharger l'application BMCI Mobile : ios : https://apps.apple.com/mr/app/bmci-mobile-mauritanie/id1467188896, android : https://play.google.com/store/apps/details?ID=mr.bmcimobile.www&pcampaignid=web_share",
    "Voici les lient pour telecharger l'application Masrvi : ios : https://apps.apple.com/mr/app/masrvi/id1540766096, android : https://play.google.com/store/apps/details?ID=com.tagpay.mwallet.masrvi&pcampaignid=web_share",
    "La BMCI propose plusieurs types de cautions et avals : Caution de Soumission, Caution de Bonne Exécution, Caution d’Avance de démarrage, Caution Retenue de Garantie, Caution Douanière, Autres Cautions, et Traite Avalisée.",
    "Le crédit immobilier pour l'acquisition à la BMCI a un plafond de 2 000 000 MRU, une durée maximale de 15 ans, un taux de 11%, et une commission de 1.5%.",
    "La BMCI propose plusieurs types de crédits aux particuliers : Crédit Standard, Silver, Premium, Electroménager, et les crédits pour Avance Fête, Scolaire, Vacance.",
    "La BMCI propose plusieurs types de cartes guichet : Carte Gimtel, Carte Gimtel Prépayée, Carte Visa Electron, Carte Visa Gold, et Carte Visa Prépayée.",
    "Pour un crédit de 100000 MRU sur 10 mois avec un taux d'intérêt de 1%, voici la simulation :\n\n- Montant du crédit : 100000 MRU\n- Durée : 10 mois\n- Taux d'intérêt : 1%\n\nMensualité : 23345 MRU par mois avec un intérêt de 13%.\n\nCommission : 1.5%\n\nTotal à rembourser : 233450 MRU.",
    "La BMCI dispose d’un réseau de 47 agences et 48 GABs répartis sur l’ensemble du territoire mauritanien, voici quelques agences très fréauentées : Agence centrale : A Nouakchott, Capital, Agence Ksar : A Ksar, Agence Toujounin : A Toujounin, Agence Nouadhbou, Agence Zouerrat, Agence Rosso, Agence Nouadhibou 2",
    "considérée comme une offre de services ou de produits, notamment produits bancaires ou d´assurance,",
    "émanant de la Banque Mauritanienne pour le Commerce International, ni comme une offre ou \
    la sollicitation d´une offre d´achat ou de vente de valeurs mobilières ou de tout autre produit d´investissement.",
    "La Banque Mauritanienne pour le Commerce International décline toute responsabilité quant au contenu des présentes",
    "pages et à l´utilisation qui pourrait en être faite par quiconque. La BMCI s’efforce d’assurer l’exactitude",
    "et la mise à jour des informations diffusées dans ce site, dont elle se réserve le droit de corriger",
    "Comment puis-je ouvrir un compte en ligne? Rien de plus simple. Rendez-vous sur la page « Devenir client » sur notre site.",
    "Suivez les instructions. Vous pouvez également contacter notre équipe de conseiller au siège de la BMCI.",
    "Notre service client est à votre disposition 0022247298000 24/7 BANQUE MAURITANIENNE POUR LE COMMERCE INTERNATIONAL Société anonyme au capital de 1 Milliard MRU Avenue de Gamal Abdel Nasser, BP: 622. Nouakchott, Maurtitanie",
    "MASRVI permet aux entreprises de gérer à distance et de manière électronique le paiement des salaires de leurs employés, offrant ainsi un gain de temps et une gestion simplifiée.",
    "Le crédit immobilier pour la construction chez la BMCI a un plafond de 3 000 000 MRU, une durée maximale de 15 ans, un taux de 11%, et une commission de 1.5%.",
    "Le DAT (Dépôt à Terme) est un produit d'épargne à durée déterminée proposé par la BMCI",
    "Pour obtenir ou demander un pret, Ouvrir l'application BMCI Mobie, connecter vous et choisissez Crédit (en bas de l'ecran), choisir 'Crédit instantanné' ou 'Nouveau crédit avec validation' et suivez les instructions.",
    "La BMCI propose plusieurs types de crédits aux particuliers : Crédit Standard, Silver, Premium, Electroménager, et les crédits pour Avance Fête, Scolaire, Vacance."
    "Pour ouvrir un compte MASRVI, téléchargez l'application depuis le Play Store ou l'App Store, inscrivez votre numéro de téléphone et votre NNI, puis identifiez-vous dans une agence BMCI ou MASRVI pour activer votre compte."
]

def initialise_collection():
    start_time = time.time()
    #content1 = crawl_website(base_url="https://bmci.mr")
    content = crawl_website(base_url="https://masrvi.mr")
    end_time = time.time()
    print(f"Web scraping took {end_time - start_time:.2f} seconds")

    text_chunks = list(process_in_chunks(content))

    for chunk in process_in_chunks(content):
        text.append(chunk)

    dataset = pd.DataFrame({"text": text})
    loader = DataFrameLoader(dataset, page_content_column="text")
    documents = loader.load()
    # recreate_collection("agents")
    # print("created -------")
    # store_data_in_collection("agents", documents)
    # set_collection_indexing('agents')
    recreate_collection("clients")
    print("created -------")
    store_data_in_collection("clients", documents)
    set_collection_indexing('clients')

@app.on_event("startup")
def on_startup():
    print("Running initialise_collection() on startup")
    initialise_collection()
