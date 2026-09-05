from langchain_groq import ChatGroq
import os
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from typing import List
from fastapi.middleware.cors import CORSMiddleware
from twilio.rest import Client
from exchange import custom_prompt


from flask import Flask, request, jsonify, session
from flask_cors import CORS

groq_api_key = os.getenv("GROQ_API_KEY")
chat = ChatGroq(temperature=1, model_name="mixtral-8x7b-32768", api_key=groq_api_key)

#app = FastAPI()

# Add CORS middleware to allow requests from any origin (use this for development)
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # Allows all origins, you can specify specific domains if needed
#     allow_credentials=True,
#     allow_methods=["*"],  # Allows all HTTP methods (GET, POST, etc.)
#     allow_headers=["*"],  # Allows all headers
# )
# Define a Pydantic model for the incoming data
class UserMessage(BaseModel):
    message: str

# Define Pydantic models for incoming request data
class WhatsAppMessage(BaseModel):
    Body: str
    From: str

# In-memory session variable (you could replace this with a more persistent solution)
# session = {'messages': []}

# @app.post("/chat")
# async def bot(user_message: UserMessage):
#     user_query = user_message.message.strip()
#     if not user_query:
#         return JSONResponse(content={"response": "Prompt is empty, please provide a valid input."}, status_code=400)

#     # Append user message to session
#     session['messages'].append({"sender": "user", "content": user_query})
    
#     # Get bot's response
#     response = custom_prompt(user_query, chat)
    
#     # Append bot message to session
#     session['messages'].append({"sender": "bot", "content": response})

#     return {"response": response, "history": session['messages']}

# Set up Twilio Credentials
from twilio.rest import Client
from twilio.twiml.messaging_response import MessagingResponse

import os
from twilio.rest import Client

def send_twilio_message(to_whatsapp, message):
    account_sid = 'AC84cb56c0334678181e6db0c22c91a811'
    auth_token = '6ebcade30502f90af29114236548ed4c'
    client = Client(account_sid, auth_token)

    from_whatsapp_number = 'whatsapp:+14155238886'
    to_whatsapp_number = to_whatsapp

    client.messages.create(
        body=message,
        from_=from_whatsapp_number,
        to = to_whatsapp_number
    )

app = Flask(__name__)
app.secret_key = "Q93zNv7oBVurcllHasFaV1gV9JLOsobtGgaLeO3c+z9"  # Needed for session management

# Define CORS rules for specific routes
CORS(app, resources={
    r"/chat": {"origins": "*"},       # Allow all origins for /chat route
    r"/whatsapp": {"origins": "*"}   # Allow all origins for /whatsapp route
})
# Initialize session data
@app.before_request
def initialize_session():
    if 'messages' not in session:
        session['messages'] = []

# Route for chat
@app.route('/chat', methods=['POST'])
def bot():
    user_query = request.json.get("message", "").strip()
    if not user_query:
        return jsonify({"response": "Prompt is empty, please provide a valid input."})

    session['messages'].append({"sender": "user", "content": user_query})
    response = custom_prompt(user_query, chat)
    session['messages'].append({"sender": "bot", "content": response})

    return jsonify({"response": response, "history": session['messages']})


# Route to handle incoming WhatsApp messages
@app.route('/whatsapp', methods=['POST', 'GET'])
def whatsapp():
    user_message = request.form.get('Body', '')
    sender_whatsapp = request.form.get('From', '')
    # Handle response
    response_message = custom_prompt(user_message, chat)
    send_twilio_message(sender_whatsapp, response_message)
    return 'Message received', 200

# Route to clear history
@app.route('/clear', methods=['POST'])
def clear_history():
    session.clear()
    return jsonify({"response": "Chat history cleared."})

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True, port=5002)