from datetime import datetime,timedelta
import json
from typing import Dict, List
from langchain.schema import (
    HumanMessage,
    SystemMessage
)
from qdrant_local import client_qdrant
import tiktoken

from langchain_core.documents import Document
from utils import *
from db.db import get_etiquettes
from sentence_transformers import CrossEncoder
from langchain_openai import ChatOpenAI

reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')    #cohere 

# Count number of tokens in a string (to respect model's input limits)
def num_tokens_from_string(string: str, model_name: str = MODEL_NAME) -> int:
    encoding = tiktoken.encoding_for_model(model_name)
    return len(encoding.encode(string))

from qdrant_client import models
from qdrant_client.models import FieldCondition

def custom_prompt(query: str,  history: List, collection_name: str, isEval: bool , chat = ChatOpenAI(model='gpt-4o-mini'),):
    """
    Custom prompt function to handle user queries, retrieve relevant documents,
    rerank them, and construct a prompt for the language model.
    
    Args:
        query (str): The user's query.
        chat: The chat model to interact with.
        history (List): The conversation history.
        collection_name (str): The name of the Qdrant collection to search.
        isEval (bool): Flag indicating if this is an evaluation run.

    Returns:
        dict: Contains the response from the model, retrieval results, top score, and context chunks.
    """
    
    # Step 1: Embed the user query to convert it into a vector
    query_vector = openai_embeddings.embed_query(query)    

    # Define a filter on documents 
    query_filter = get_Query_filter() 
    # query_filter = None 
    

    # Step 2: Perform a similarity search in Qdrant using the native client
    retrieval_results = client_qdrant.search(
    collection_name=collection_name,
    query_vector=query_vector,
    query_filter=query_filter ,
    limit=20,
    with_payload=True
    )

    print(f"search results : {len(retrieval_results)}".center(50, "-"))
    

    top_score = retrieval_results[0].score if retrieval_results else 0.0

    # Step 3: Format the retrieved documents
    results = []
    for result in retrieval_results:
        payload = result.payload or {}
        content = payload.get("page_content", "")
        content= result.payload.get('page_content', None) if result.payload else None


        # Only keep non-empty text chunks
        if isinstance(content, str) and content.strip():
            results.append(Document(page_content=content, metadata=payload))
        else:
            print("Skipping invalid or empty document:")


    # Reranking 
    context_chunks , current_messages = rerank_and_prepare_context(
        query=query, 
        results=results, 
        reranker=reranker, 
        history=history,
        top_k=5
    )


    # Step 4: Reduce the prompt size if it exceeds the token limit
    while True:
        source_knowledge = "\n".join(context_chunks)
        
        # Construct augmented prompt using context and chat history
        augment_prompt = f"""Using the contexts below, and the messages history answer the query, try to detect the language and answer accordingly only in one language, be consise and to the point.
        Contexts:
        {source_knowledge}

        Messages:
        {[message.content for message in current_messages]}

        Score: {[top_score]}
        Query: {query}
        """

        token_count = num_tokens_from_string(augment_prompt)
        # print(f"Token count: {token_count}")

        # If within token limit, proceed
        if token_count < MAX_TOKENS:
            break

        # Otherwise, reduce size by trimming context or history
        if len(current_messages) > 1 :
            current_messages = current_messages[1:]
        else:
            raise ValueError("Prompt cannot be reduced below limit.")

    # Step 5: Append final query message to history
    prompt = HumanMessage(content=augment_prompt)
    current_messages.append(prompt)

    # Step 6: Query the language model with the constructed message history
    res = chat.invoke(current_messages)

    # Step 7: Clean up formatting and return the response
    response_text = res.content.replace("<", "").replace(">", "")
    # Extra processing for title and intention
    
    user_messages = [m for m in history if isinstance(m, HumanMessage)]

    if len(user_messages) > 2 or isEval:
        conversation_title = ""
        predicted_intention = ""
        print("predicting title and etiquette")
        etiquette_labels = get_etiquettes()
        etiquette_string = ", ".join(etiquette_labels)

        prediction_prompt = f"""
        Given the conversation below, generate a short and meaningful title (3-6 words) and choose one intention from the following list (of names and description, seperated by :) that best matches the user's goal, return only the name of the intention: [{etiquette_string}].

        Conversation:
        {query}

        Answer as JSON:
        {{
          "title": "...",
          "intention": "..."
        }}
        """
        
        prediction_result = chat.invoke([HumanMessage(content=prediction_prompt)])
        try:
            import json
            parsed = json.loads(prediction_result.content)
            conversation_title = parsed.get("title", "")
            predicted_intention = parsed.get("intention", "")
        except Exception as e:
            print("Failed to parse prediction:", e)

    return {
        "title": conversation_title,
        "intention": predicted_intention,
        "response": response_text,
        "retrieval_results": retrieval_results,  # Add retrieval results
        "top_score": top_score,                 # Add top score
        "contexts": context_chunks              # Add context chunks
        }
    


# -------------------------------------------------------------------------------------------------------------
from typing import List, Tuple, Any, Optional
import logging
logger = logging.getLogger(f"RANKER {reranker.__class__.__name__} ")
logging.basicConfig(level=logging.INFO)


def rerank_and_prepare_context(
    query: str, 
    results: List[Any], 
    reranker: Any, 
    history: List[Any],
    top_k: int = 3
) -> Tuple[List[str], List[Any]]:
    """
    Rerank search results and prepare context for chat.
    
    Args:
        query: Search query string
        results: List of document objects with page_content attribute
        reranker: Reranker model instance with predict method
        history: Chat history to preserve
        top_k: Number of top documents to return (default: 3)
    
    Returns:
        Tuple of (context_chunks, current_messages)
        - context_chunks: List of top document contents
        - current_messages: Copy of chat history
    
    Raises:
        ValueError: If top_k is not positive
        AttributeError: If documents don't have page_content attribute
    """
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    
    # Early return for empty results
    if not results:
        logger.info("No results found, skipping reranking")
        return [], history.copy()
    
    logger.info(f"Reranking {len(results)} results")
    
    try:
        # Validate documents have required attribute
        if not all(hasattr(doc, 'page_content') for doc in results):
            raise AttributeError("All documents must have 'page_content' attribute")
        
        # Create query-document pairs for reranking
        pairs = [(query, doc.page_content) for doc in results]
        
        # Get reranking scores
        scores = reranker.predict(pairs)
        
        # Combine documents with scores and sort by score (descending)
        scored_docs = sorted(
            zip(results, scores), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        # Extract top documents
        top_docs = [doc for doc, _ in scored_docs[:top_k]]
        
        # Extract context chunks
        context_chunks = [doc.page_content for doc in top_docs]
        
        logger.info(f"Selected top {len(top_docs)} documents after reranking")
        
    except Exception as e:
        logger.error(f"Error during reranking: {e}")
        # Fallback: return original results truncated to top_k
        context_chunks = [doc.page_content for doc in results[:top_k]]
    
    # Return context and preserved history
    return context_chunks, history.copy()



def get_Query_filter():
    """
    Define a filter for Qdrant queries to only include documents
    that have been updated in the last 30 days.
    
    Returns:
        models.Filter: The filter to apply to the query.
    """
    return models.Filter(
        should=[
            FieldCondition(
                key='last_updated',
                range=models.DatetimeRange(
                    gte=datetime.now() - timedelta(days=30),  # Only consider documents updated in the last 30 days
                    lte=datetime.now()
                )
            )
        ]
    )


from process_document import generate_summary_from_image


from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage
)

from sentence_transformers import CrossEncoder
from langchain_openai import ChatOpenAI
from models import Message, UserMessage


# Count number of tokens in a string (to respect model's input limits)
def num_tokens_from_string(string: str, model_name: str = MODEL_NAME) -> int:
    encoding = tiktoken.encoding_for_model(model_name)
    return len(encoding.encode(string))

# Method 1: Core response generation
def get_response(
    query: str,
    chat,
    history: List[Message],
    collection_name: str
) -> Dict[str, Any]:
    # Embed query
    query_vector = openai_embeddings.embed_query(query)
    
    # Retrieve context
    search_results = client_qdrant.search(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=Qdrant_limit,
        with_payload=True
    )
    
    # Format results
    results = []
    for result in search_results:
        if payload := result.payload:
            if content := payload.get("page_content", "").strip():
                results.append(Document(page_content=content, metadata=payload))
    
    # Rerank and select top docs
    if results:
        pairs = [(query, doc.page_content) for doc in results]
        scores = reranker.predict(pairs)
        scored_docs = sorted(zip(results, scores), key=lambda x: x[1], reverse=True)
        top_docs = [doc for doc, _ in scored_docs[:3]]
    else:
        top_docs = []
    
    context_chunks = [doc.page_content for doc in top_docs]
    augmented_prompt = build_augmented_prompt(query, context_chunks, history)
    
    # Get LLM response
    prompt_msg = HumanMessage(content=augmented_prompt)
    messages = history + [prompt_msg]
    response = chat.invoke(messages).content
    
    return {
        "response": response.strip("<>"),
        "contexts": context_chunks,
        "retrieval_results": search_results,
        "top_score": search_results[0].score if search_results else 0.0
    }

# Method 2: Title/intention prediction (call only when needed)
from typing import Dict, List
import json
import logging
from langchain.schema import HumanMessage  # assuming this import is used

def get_conversation_title_and_intention(
    query: str,
    history: List,
    chat,
    etiquette_labels: List[str]
) -> Dict[str, str]:
    """
    Generates a short title and identifies both the category and intention of a conversation
    based on provided etiquette labels.

    Args:
        query (str): The conversation history or user message.
        chat: The LLM chat interface (e.g., LangChain chat object).
        etiquette_labels (List[str]): List of labeled categories with their descriptions and intentions.

    Returns:
        Dict[str, str]: Dictionary containing 'title', 'category', and 'intention'.
    """

    etiquette_str = ":".join(etiquette_labels)
    logger = logging.getLogger("RAGAS-Evaluator")

    prompt = f"""
Given the conversation below, generate a short and meaningful title (3-6 words) and choose the most relevant **category** and **intention** from the following list. Each item is a category name followed by a description, and includes a list of intentions. Return both the matched category name and the intention name.

Conversation:
{history}

Intention categories (with descriptions and contained intentions):
[{etiquette_str}]

Return JSON:
{{ 
  "title": "...", 
  "category": "...", 
  "intention": "..." 
}}
"""

    try:
        result = chat.invoke([HumanMessage(content=prompt)])
        parsed = json.loads(result.content)

        return {
            "title": parsed.get("title", ""),
            "category": parsed.get("category", ""),
            "intention": parsed.get("intention", "")
        }

    except Exception as e:
        logger.warning("Failed to parse model output: %s", e)
        return {
            "title": "",
            "category": "",
            "intention": ""
        }



def get_title_and_intention(
    query: str,
    chat,
    etiquette_labels: List[str]
) -> Dict[str, str]:
    """
    Generates a short title and identifies both the category and intention of a conversation
    based on provided etiquette labels.

    Args:
        query (str): The conversation history or user message.
        chat: The LLM chat interface (e.g., LangChain chat object).
        etiquette_labels (List[str]): List of labeled categories with their descriptions and intentions.

    Returns:
        Dict[str, str]: Dictionary containing 'title', 'category', and 'intention'.
    """

    etiquette_str = ":".join(etiquette_labels)
    logger = logging.getLogger("RAGAS-Evaluator")

    prompt = f"""
Given the conversation below, generate a short and meaningful title (3-6 words) and choose the most relevant **category** and **intention** from the following list. Each item is a category name followed by a description, and includes a list of intentions. Return both the matched category name and the intention name.

Conversation:
{query}

Intention categories (with descriptions and contained intentions):
[{etiquette_str}]

Return JSON:
{{ 
  "title": "...", 
  "category": "...", 
  "intention": "..." 
}}
"""

    try:
        result = chat.invoke([HumanMessage(content=prompt)])
        parsed = json.loads(result.content)

        return {
            "title": parsed.get("title", ""),
            "category": parsed.get("category", ""),
            "intention": parsed.get("intention", "")
        }

    except Exception as e:
        logger.warning("Failed to parse model output: %s", e)
        return {
            "title": "",
            "category": "",
            "intention": ""
        }

# Helper: Build context-aware prompt
def build_augmented_prompt(query, contexts, history):
    source_knowledge = "\n".join(contexts)
    history_str = "\n".join([msg.content for msg in history])
    
    prompt = f"""
    Using contexts and history below, answer the query:
    
    Contexts:
    {source_knowledge}
    
    History:
    {history_str}
    
    Query: {query}
    """
    # Add token trimming logic here if needed
    return prompt


# Predefined system prompt
SYSTEM_PROMPT = (
    "You are a helpful assistant ONLY for the clients and agents of the Banque Mauritanienne "
    "pour le Commerce International (BMCI) and the clients and agents of Masrvi. "
    "Your answer should be between 200 and 500 characters. "
    "Always respond like an employee of BMCI."
)