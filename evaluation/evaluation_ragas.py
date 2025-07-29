from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import json
import logging
import os
import sys
import time
from typing import Any, Callable, List, Dict, Optional, Union
from langchain.document_loaders import DataFrameLoader
from qdrant_client.http.models import ScoredPoint
import multiprocessing
from multiprocessing import Pool
from functools import partial

from threading import Lock



from datasets import Dataset
import pandas as pd
from ragas import evaluate
from ragas.metrics import (
    faithfulness, 
    answer_relevancy, 
    context_precision, 
    context_recall, 
    answer_correctness, 
    answer_similarity, 
    answer_relevancy
)
import logging

import tiktoken

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from qdrant_local import recreate_collection, set_collection_indexing, store_data_in_collection

from models import Message, UserMessage


class Evaluator () :
    def __init__(self):
        """
        - question
        - answer (chatbot response)
        - contexts (retrieved chunks)
        - ground_truth (optional)
        """
        self.data : List[Dict] = []
        self.report : Dict = {}
        
        self.logger = logging.getLogger("RAGAS-Evaluator")
        logging.basicConfig(level=logging.INFO)
        self.total =  0




    def process_data(self , file_path:str, column_mapping:Dict[str,str] ,max_rows:int = 100,condition: Callable[[Dict[str, str]], bool] = None) -> List[Dataset]:
        temp = load_test_cases_from_excel(
            file_path=file_path,
            column_mapping=column_mapping,
            max_rows=max_rows,
            condition=condition
            )
        self.total = len(temp)
        for i, row in enumerate(temp, start=1):
            print(f"Processing test_case {i}")
            timer_start = time.time()
            try:
                # response = get_model_response_no_api(user_input=row["question"], current_messages=[],category= row.get('category', None))
                # Step 1: Get response (without title/intention)
                response = get_response(
                    query=row["question"],
                    chat=chat_openai,
                    history=[SystemMessage(content=SYSTEM_PROMPT)],
                    collection_name="clients"
                )
                
                # Step 2: Get title/intention ONLY for evaluation
                etiquette_labels = get_etiquettes()  # Fetch once globally if possible!
                title_intention = get_title_and_intention(
                    query=row["question"],
                    chat=chat_openai,
                    etiquette_labels=etiquette_labels
                )
                logging.info(f"Category for question {i}: {title_intention.get('category', 'N/A')}")
                if response['response'] is None:
                    raise ValueError("Model response is None")
                row['answer'] = response['response']
                row['botCategory'] = title_intention.get('category', 'N/A')  # Add category if available
                row['title'] = title_intention.get('title', 'N/A')  #
                row['bot_intention'] = title_intention.get('intention', 'N/A')  # Add intention if available
                row['contexts'] = response.get('contexts', [])  # Add contexts if available
                row["retrieval_results"] = process_retrieval_results(response.get('retrieval_results', []))  # Add retrieval results if available
                row['top_retrieval_score'] = response.get('top_score', 0.0)  # Add top retrieval score if available
            except Exception as e:
                self.logger.error(f"Failed to process question {i}: {e}")
                row['answer'] = None  # Or some default fallback
                row['error'] = str(e)  # Store the error message in the row
                continue
            
            timer_end = time.time()
            row['processing_time'] = timer_end - timer_start
            self.data.append(row)
            self.logger.info(f"Processed {i}/{self.total} test cases".center(50, "-"))





    def _evaluate_single_test_case(self, test_case, rag_metrics):
        """Helper function to evaluate a single test case for multiprocessing."""

        ragas_test = [
            {
                "question": test_case['question'],
                "answer": test_case['answer'],
                "contexts": test_case.get("contexts", []),
                "ground_truth": test_case["ground_truth"] if 'ground_truth' in test_case else None,
                'response' : test_case["ground_truth"] if 'ground_truth' in test_case else None,
            }
        ]

        for item in ragas_test:
            if isinstance(item['ground_truth'], list) and item['ground_truth']:
                # Join all ground truth strings with a separator (e.g., newline)
                item['ground_truth'] = "\n".join(str(gt) for gt in item['ground_truth'])
                item['response'] = "\n".join(str(gt) for gt in item['response'])
            elif not isinstance(item['ground_truth'], str):
                item['ground_truth'] = str(item['ground_truth'])
                item['response'] = str(item['response'])

        ragas_test_case = Dataset.from_list(ragas_test)

        try:
            scores = evaluate(
                ragas_test_case,
                rag_metrics,
            ).scores
            return {"index": self.data.index(test_case), "scores": scores} # Return index to update original data
        except Exception as e:
            self.logger.error(f"Error evaluating test case: {test_case['question']} - {e}")
            return {"index": self.data.index(test_case), "scores": None}
        

    
    def run_evaluation(self, rag: bool = False):
        """Run RAGAS evaluation using selected metrics - OPTIMIZED VERSION"""
        if self.data is None:
            self.logger.error("No test data found. Please process data first.")
            self.logger.info("processing data...")
            self.process_data()
            self.logger.info("Data processing complete.")

        # Define metrics to use based on available data
        rag_metrics = [
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,  # Requires ground_truth
            answer_correctness,  # Requires ground_truth and embeddings
            answer_similarity  # Requires ground_truth and embeddings
        ]

        # OPTIMIZATION 1: Batch process all data at once instead of one by one
        ragas_test_list = []
        
        for i, test_case in enumerate(self.data, start=1):
            # Prepare ground truth
            ground_truth = test_case.get("ground_truth")
            if isinstance(ground_truth, list) and ground_truth:
                ground_truth = "\n".join(str(gt) for gt in ground_truth)
            elif ground_truth is not None:
                ground_truth = str(ground_truth)
            
            ragas_item = {
                "question": test_case['question'],
                "answer": test_case['answer'],
                "contexts": test_case.get("contexts", []),
                "ground_truth": ground_truth,
                "response": ground_truth,  # Assuming this should be the same as ground_truth
            }
            ragas_test_list.append(ragas_item)
        
        # Create single dataset from all test cases
        ragas_dataset = Dataset.from_list(ragas_test_list)
        
        print(f"\nStarting Ragas evaluation for {len(self.data)} test cases".center(60, "-"))
        
        # OPTIMIZATION 2: Single batch evaluation
        results = evaluate(ragas_dataset, rag_metrics)
        
        # OPTIMIZATION 3: Assign scores back to individual test cases
        # Handle different RAGAS result formats
        if hasattr(results, 'scores'):
            scores_data = results.scores
        else:
            scores_data = results
        
        for i, test_case in enumerate(self.data):
            if isinstance(scores_data, dict):
                # Dictionary format: {metric_name: [score1, score2, ...]}
                test_case['scores'] = {
                    metric: scores_data[metric][i] for metric in scores_data.keys()
                }
            elif isinstance(scores_data, list):
                # List format: [score1, score2, ...]
                test_case['scores'] = scores_data[i] if i < len(scores_data) else None
            else:
                # DataFrame format
                try:
                    test_case['scores'] = scores_data.iloc[i].to_dict()
                except:
                    test_case['scores'] = scores_data
        
        self.logger.info("All test cases evaluated successfully.".center(60, "-"))
    
    def get_response(self) :
        """
        Analyzes the provided response object from a test evaluation and generates a summary report.

        Returns:
            Dict: A summarized report with overall statistics and a breakdown by category,
                  including a list of poor-performing cases.
        """
        test_details = []
        by_category_stats = {}

        for test_case in self.data:
            retrieved_contexts = [
                d.get('page_content', "") for d in test_case.get('retrieval_results', [])
                if d.get('page_content') is not None
            ]

            test_case_details = {
                "userInput": test_case['question'],
                "expectedResponse": test_case.get('ground_truth', "N/A"),
                "actualResponse": test_case.get('answer', "N/A"),
                "scores": test_case.get('scores', {}),  # Expecting a dict now
                "intention": test_case.get('intention', 'N/A'),
                "category": test_case.get('category', 'N/A'),
                "botCategory": test_case.get('botCategory', 'N/A'),
                "botIntention": test_case.get('bot_intention', 'N/A'),
                "exactMatch": 1 if test_case.get('is_exact_possible', False) else 0,
                "processingTime": test_case.get('processing_time', 0.0),
                "topRetrievalScore": test_case.get('top_retrieval_score', 0.0),
                "retrievedContexts": retrieved_contexts,
                "error": test_case.get('error', "null"),
                "evaluation_scores": test_case.get('scores', {}), # Keep for redundancy/consistency
            }
            test_details.append(test_case_details)

            category = test_case.get('category', 'N/A')
            if category not in by_category_stats:
                by_category_stats[category] = {
                    "count": 0,
                    "successful": 0,
                    "avgProcessingTime": 0.0,
                    "avgRetrievalScore": 0.0,
                    "scores_sum": {},
                    "scores_count": {}
                }

            category_stats = by_category_stats[category]
            category_stats["count"] += 1
            if test_case.get('answer') is not None:
                category_stats["successful"] += 1
            
            category_stats["avgProcessingTime"] += test_case.get('processing_time', 0.0)
            
            # Sum up retrieval scores for category average
            retrieval_scores_for_case = [d.get('score', 0.0) for d in test_case.get('retrieval_results', [])]
            if retrieval_scores_for_case:
                category_stats["avgRetrievalScore"] += sum(retrieval_scores_for_case) / len(retrieval_scores_for_case)
            
            # Aggregate RAGAS scores by category
            scores = test_case.get('scores', {})
            if isinstance(scores, dict):
                for key, score in scores.items():
                    if isinstance(score, (int, float)):
                        category_stats["scores_sum"][key] = category_stats["scores_sum"].get(key, 0.0) + score
                        category_stats["scores_count"][key] = category_stats["scores_count"].get(key, 0) + 1
            
        # Calculate averages for by_category_stats
        for category, stats in by_category_stats.items():
            if stats["count"] > 0:
                if stats["successful"] > 0:
                    stats["avgRetrievalScore"] /= stats["successful"]
                else:
                    stats["avgRetrievalScore"] = 0.0

                for metric, total_score in stats["scores_sum"].items():
                    count = stats["scores_count"].get(metric, 0)
                    stats[f"avg_{metric}"] = total_score / count if count > 0 else 0.0
                del stats["scores_sum"]
                del stats["scores_count"]
            
        poor_performers = [d for d in get_lowest_score_test_cases(self.data).values()]

        total_retrieval_score_sum = sum(sum(d.get('score', 0.0) for d in tc.get('retrieval_results', [])) for tc in self.data)

        total_retrieval_results_count = sum(len(tc.get('retrieval_results', [])) for tc in self.data)
        avg_overall_retrieval_score = (
            total_retrieval_score_sum / total_retrieval_results_count 
            if total_retrieval_results_count > 0 
            else 0.0
        )
        
        report = {
            "totalTests": len(self.data),
            "errorCount": len([d for d in self.data if d.get('error')]),
            "avgScores": calculate_average_scores(self.data), # This function is expected to handle the dict scores
            "avgProcessingTime": sum(d['processing_time'] for d in self.data if 'processing_time' in d) / len(self.data) if self.data else 0.0,
            "avgRetrievalScore": avg_overall_retrieval_score,
            "contextRecallRate": "null", # Placeholder as per original
            "byCategory": by_category_stats,
            "details": test_details,
            "poorPerformingCases": poor_performers
        }

        self.report = report
        self.report = sanitize_floats(report)
        return report

    
    def print_results(self):
        """Pretty-print RAGAS evaluation results - FIXED VERSION"""
        if self.data is None:
            self.logger.info("running evaluation...")
            self.logger.info("Evaluation complete.")
            return  

        self.logger.info("Retrieval Results:".center(50, "-"))
        mean_top_retrieval_score = sum(d.get('top_retrieval_score', 0.0) for d in self.data) / len(self.data) if self.data else 0.0
        min_top_retrieval_score = min(d.get('top_retrieval_score', 0.0) for d in self.data) if self.data else 0.0
        max_top_retrieval_score = max(d.get('top_retrieval_score', 0.0) for d in self.data) if self.data else 0.0
        print(f"  ➤ Average Top Retrieval Score: {mean_top_retrieval_score:.4f}")
        print(f"  ➤ Minimum Top Retrieval Score: {min_top_retrieval_score:.4f}")
        print(f"  ➤ Maximum Top Retrieval Score: {max_top_retrieval_score:.4f}")

        self.logger.info("RAGAS Evaluation Results:")

        all_scores = {}
        for d in self.data:
            # Handle different score formats
            scores = d.get('scores', {})
            
            # Debug: Check what format the scores are in
            if not scores:
                continue
                
            # Handle different score formats
            if isinstance(scores, dict):
                # Format: {'metric1': score1, 'metric2': score2, ...}
                for metric, score in scores.items():
                    if metric not in all_scores:
                        all_scores[metric] = []
                    all_scores[metric].append(score)
            elif isinstance(scores, list) and len(scores) > 0:
                # Format: [{'metric1': score1, 'metric2': score2, ...}]
                if isinstance(scores[0], dict):
                    for metric, score in scores[0].items():
                        if metric not in all_scores:
                            all_scores[metric] = []
                        all_scores[metric].append(score)
                else:
                    # Format: [score1, score2, ...]
                    # This case needs metric names, you might need to adjust based on your metrics
                    metric_names = ['faithfulness', 'answer_relevancy', 'context_precision', 
                                'context_recall', 'answer_correctness', 'answer_similarity']
                    for i, score in enumerate(scores):
                        if i < len(metric_names):
                            metric = metric_names[i]
                            if metric not in all_scores:
                                all_scores[metric] = []
                            all_scores[metric].append(score)
            else:
                # Handle other formats or single score
                print(f"  ➤ Unexpected score format for test case: {type(scores)}")

        if all_scores:
            print("\nSummary Stats:".center(50, "-"))
            for metric, scores in all_scores.items():
                if scores:  # Make sure scores list is not empty
                    print(f"  ➤ {metric:<20} ➜ Count: {len(scores):<3} | Avg: {sum(scores)/len(scores):.4f} | Min: {min(scores):.4f} | Max: {max(scores):.4f}")
        else:
            print("  ➤ No scores available for summary statistics.")
            # Debug: Print what we actually have
            print("\nDebug - Score formats found:")
            for i, d in enumerate(self.data[:3]):  # Show first 3 for debugging
                print(f"  Test case {i}: scores = {d.get('scores', 'No scores')}")

        # Ensure there are successful response times before attempting calculations
        successful_response_times = [d['processing_time'] for d in self.data if 'processing_time' in d]
        if successful_response_times:
            print(f"  ➤ processing_time_avg : {sum(successful_response_times) / len(successful_response_times):.4f} seconds")
            print(f"  ➤ processing_time_min : {min(successful_response_times):.4f} seconds")
            print(f"  ➤ processing_time_max : {max(successful_response_times):.4f} seconds")
        else:
            print("  ➤ No successful response times to display.")



    def evaluate(self,file_path:str = None, column_mapping:Dict[str,str] = None ,max_rows:int = 100, rag:bool=False,condition: Callable[[Dict[str, str]], bool] = None):
        """Main method to run all steps"""
        if file_path is not None and column_mapping is not None:
            self.process_data(
                file_path=file_path,
                column_mapping=column_mapping,
                max_rows=max_rows,
                condition=condition
            )

        self.run_evaluation(rag=rag)
        report = self.get_response()
        self.print_results()
        return report



    def generate_report(self):
        report = {}
        report["totalTests"] = len(self.data)
        report["successfulTests"] = len([d for d in self.data if d.get('answer') is not None])

    




# --------------------------------------------------------------------------

#utils functions : 
        

def calculate_average_scores(test_cases: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Calculates the average of each score type across all test cases.
    """
    score_sums = {}
    score_counts = {}

    for test_case in test_cases:
        # Expecting scores to be a list containing a single dictionary
        scores_list = test_case.get("scores", [])
        if not scores_list or not isinstance(scores_list, dict): # Adjusted to expect dict directly
            continue  

        scores = scores_list  # Use scores directly if it's a dict

        for score_name, score_value in scores.items():
            if isinstance(score_value, (int, float)): # Ensure score_value is numeric
                score_sums[score_name] = score_sums.get(score_name, 0.0) + score_value
                score_counts[score_name] = score_counts.get(score_name, 0) + 1

    average_scores = {}
    for score_name in score_sums:
        count = score_counts.get(score_name, 0)
        # Guard against division by zero
        average_scores[score_name] = score_sums[score_name] / count if count > 0 else 0.0

    return average_scores
        

def get_lowest_score_test_cases(test_cases: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Returns a dictionary where each key is a score name and the value is the test case
    (dict) that has the lowest value for that score.
    """
    lowest_scores = {}

    for test_case in test_cases:
        # Expecting scores to be a single dictionary
        scores = test_case.get("scores", {})
        if not scores or not isinstance(scores, dict):
            continue

        for score_name, score_value in scores.items():
            if isinstance(score_value, (int, float)):
                if score_name not in lowest_scores or score_value < lowest_scores[score_name][1]:
                    lowest_scores[score_name] = (test_case, score_value)

    return {score: entry[0] for score, entry in lowest_scores.items()}




def process_retrieval_results(retrieval_results : List[ScoredPoint]) -> List[Dict]:
    
    """
    Traite une liste d'objets ScoredPoint de Qdrant pour extraire
    les scores et le contenu des pages.

    Args:
        input_list: Une liste d'objets ScoredPoint, tels que retournés par Qdrant,
                    chacun contenant un score et un dictionnaire 'payload' avec une clé 'page_content'.

    Returns:
        Un tuple contenant deux listes :
        - scores (List[float]): Une liste des scores extraits.
        - page_contents (List[str]): Une liste du contenu des pages extrait.
    """
    retrieval_data = []

    # Itérer sur chaque objet ScoredPoint dans la liste d'entrée
    for item in retrieval_results:
        # Extraire le score et l'ajouter à la liste des scores
        retrieval_data.append({
            'id': item.id,
            'score': item.score,
            'page_content': item.payload.get('page_content', None) if item.payload else None
        })
        

    return retrieval_data
        



def load_test_cases_from_excel(
    file_path: str,
    column_mapping: Dict[str, str],
    max_rows: int = None,
    rag: bool = False,
    condition: Callable[[Dict[str, str]], bool] = None
) -> List[Dict]:
    df = pd.read_excel(file_path)
    test_cases = []
    
    for _, row in df.iterrows():
        if max_rows and len(test_cases) >= max_rows:
            break
            
        row_dict = row.to_dict()
        if condition and not condition(row_dict):
            continue
            
        test_case = {}
        for key, col in column_mapping.items():
            val = row_dict.get(col)
            if val is None or pd.isna(val):
                test_case[key] = [] if key in ['context', 'ground_truth'] else ""
                continue
                
            if key in ['context', 'ground_truth']:
                test_case[key] = [v.strip() for v in str(val).split(",") if v.strip()]
            else:
                test_case[key] = str(val)
                
        test_cases.append(test_case)
        
    return test_cases



from langchain.schema import (
    SystemMessage,
)



from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage
)

from utils import *
from qdrant_local import client_qdrant
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


import math
def sanitize_floats(obj):
    """Recursively sanitize NaN and infinity values in dictionaries/lists"""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: sanitize_floats(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_floats(item) for item in obj]
    return obj