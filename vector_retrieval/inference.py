import os
import sys
import logging
import warnings
import time
from typing import List, Dict, Any, Tuple
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# Suppress warnings
warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

import torch
from sentence_transformers import CrossEncoder, SentenceTransformer
from pymilvus import connections, Collection, MilvusClient
from crewai import Agent, Task, Crew, Process, LLM

# Constants
TARGET_URI = "http://localhost:19530"
MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"
COLLECTION_NAME = "Trade_Finance_Article"
EMBEDDING_MODEL_NAME = "Alibaba-NLP/gte-large-en-v1.5"
RERANKING_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "0TD9nsBifR6Lkr1kOag9aikbCBImYfGg")
MISTRAL_MODEL = "mistral/mistral-large-latest"

class RerankerWrapper:
    def __init__(self, model_name: str):
        logging.info(f"Loading reranker: {model_name}")
        self.model = CrossEncoder(model_name)
    
    def rerank(self, query: str, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not chunks:
            return []
        pairs = [[query, chunk['text']] for chunk in chunks]
        scores = self.model.predict(pairs)
        for i, chunk in enumerate(chunks):
            chunk['rerank_score'] = float(scores[i])
        sorted_chunks = sorted(chunks, key=lambda x: x['rerank_score'], reverse=True)
        return sorted_chunks

class MilvusSearch:
    def __init__(self):
        self._local = threading.local()
    
    def milvus_search_caller(self, query_to_execute, metadata_filter, models, topk):
        results = self.vector_search(
            models['embedding_model'],
            query_to_execute,
            models['vectordb_collection'], 
            top_k=topk, 
            metadata_filter=metadata_filter
        )
        return results

    def vector_search(self, embedding_model, query, collection, top_k, metadata_filter, anns_field: str = "vector"):
        dense_query = embedding_model.encode(query).tolist()
        try:
             index_param = collection.index().to_dict()["index_param"]
             index_type = index_param.get('metric_type', 'COSINE')
        except Exception:
             index_type = "COSINE"
        
        search_params = {"metric_type": index_type, "params": {"ef": 100}}
        
        if metadata_filter:     
            logging.info(f"Running with metadata filter: {metadata_filter}")
            results = collection.search(
                data=dense_query,
                anns_field=anns_field,
                param=search_params,
                limit=top_k,
                output_fields=['text', 'metadata'],
                consistency_level="Strong",
                expr=metadata_filter
            )
        else:
            logging.info("Running without metadata filter")
            results = collection.search(
                data=dense_query,
                anns_field=anns_field,
                param=search_params,
                limit=top_k,
                output_fields=['text', 'metadata'],
                consistency_level="Strong"
            )

        search_result = []
        for res in results:
            for hit in res:
                search_result.append({
                    'retrieval_score': float(hit.distance),
                    'text': hit.entity.text,
                    'metadata': hit.entity.metadata
                })
        return search_result

class Retriever:
    def __init__(self, max_workers: int = None):
        self.search_caller = MilvusSearch()
        self.max_workers = max_workers

    def search_and_rerank(self, models: Dict, collection_name: str, filter_query: str, search_topk: int, user_query: str) -> Tuple[str, str, List[Dict]]:
        try:
            local_models = models.copy()
            local_models["vectordb_collection"] = models["connected_collection"][collection_name]
            
            start_time = time.time()
            results_k = self.search_caller.milvus_search_caller([user_query], filter_query, local_models, search_topk)
            end_time = time.time()
            
            logging.info(f"Collection: {collection_name}, Filter: '{filter_query}', Results: {len(results_k)}, Time: {end_time - start_time:.2f}s")
            
            return collection_name, filter_query, results_k
            
        except Exception as e:
            logging.error(f"Error searching collection {collection_name} with filter '{filter_query}': {str(e)}")
            return collection_name, filter_query, []

    def _process_collections_parallel(self, models: Dict, collections_dict: Dict, user_query: str, chunks_per_collection: int, stage_name: str) -> List[Dict]:
        search_tasks = []
        for collection_name, metadata_filter_list in collections_dict.items():
            logging.info(f"{collection_name}: processing collection")
            if metadata_filter_list and metadata_filter_list[0]:
                chunks_per_filter = chunks_per_collection // len(metadata_filter_list)
                remaining_chunks = chunks_per_collection % len(metadata_filter_list)
                for i, metadata_filter in enumerate(metadata_filter_list):
                    current_chunks = chunks_per_filter + (1 if i < remaining_chunks else 0)
                    search_tasks.append((collection_name, metadata_filter, current_chunks))
            else:
                search_tasks.append((collection_name, "", chunks_per_collection))
        
        logging.info(f"{stage_name}: Starting parallel retrieval for {len(search_tasks)} tasks")
        
        retrieved_data = []
        completed_tasks = 0
        failed_tasks = 0
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_task = {}
            for collection_name, metadata_filter, chunk_count in search_tasks:
                future = executor.submit(
                    self.search_and_rerank, 
                    models, 
                    collection_name, 
                    metadata_filter, 
                    chunk_count, 
                    user_query
                )
                future_to_task[future] = (collection_name, metadata_filter, chunk_count)
            
            for future in as_completed(future_to_task):
                collection_name, metadata_filter, chunk_count = future_to_task[future]
                try:
                    result_collection, result_filter, results = future.result()
                    retrieved_data.extend(results)
                    completed_tasks += 1
                except Exception as e:
                    failed_tasks += 1
                    logging.error(f"{stage_name} - Task failed: {str(e)}")
        
        return retrieved_data

    def run_query_manager(self, models: Dict) -> Dict:
        """
        Simplified retrieval: Single stage.
        Retrieve -> Rerank -> Top K.
        No Alpha/Beta/Gamma classification.
        """
        start_time = time.time()
        user_query = models["QUERY"]
        top_k = int(models["TOP_K"])
        
        # Get all collections from connected_collection
        connected = models["connected_collection"]
        
        # Assume no filters for simplification, or default to empty string
        collections_to_search = {name: [""] for name in connected.keys()}
        
        logging.info(f"Starting single-stage retrieval for {len(collections_to_search)} collections...")
        
        # Retrieve chunks (fetching top_k candidates directly)
        raw_results = self._process_collections_parallel(
            models, 
            collections_to_search, 
            user_query, 
            chunks_per_collection=top_k, 
            stage_name="Retrieval"
        )
        
        # Rerank
        final_results = []
        if raw_results:
            logging.info(f"Reranking {len(raw_results)} chunks...")
            reranked = models["reranking_model"].rerank(query=user_query, chunks=raw_results)
            final_results = reranked[:top_k]
            logging.info(f"Reranking done. Top {len(final_results)} selected.")
            
        end_result = {
            "query": user_query,
            models["source"]: final_results
        }
        
        logging.info(f"Total processing time: {time.time() - start_time:.2f}s")
        return end_result

def generate_answer_crew(query: str, context_chunks: List[Dict[str, Any]]) -> str:
    if not context_chunks:
        return "No relevant context found to answer the query."
        
    context_text = ""
    for i, c in enumerate(context_chunks):
        context_text += f"Source {i+1}:\n{c['text']}\n\n"
    
    llm = LLM(model=MISTRAL_MODEL, api_key=MISTRAL_API_KEY)
    
    agent = Agent(
        role='Trade Finance Expert',
        goal='Answer user questions accurately based on the provided partial context.',
        backstory='You are a specialized assistant for Trade Finance. You answer strictly based on the context provided.',
        llm=llm,
        verbose=False
    )
    
    task_desc = f"""
    You are provided with retrieved context from Trade Finance documents.
    Answer the user's question based strictly on this context. 
    If the answer is not in the context, state that you do not have enough information.
    
    Context:
    {context_text}
    
    User Question: {query}
    
    Provide a clear, concise, and comprehensive answer.
    """
    
    task = Task(
        description=task_desc,
        expected_output="The final answer text.",
        agent=agent
    )
    
    crew = Crew(
        agents=[agent],
        tasks=[task],
        process=Process.sequential
    )
    
    result = crew.kickoff()
    return str(result)

def main():
    # Hardcoded query as preferred
    user_query = "Explain the rules governing if any change in the guarantee ?"
    
    logging.info(f"Received query: {user_query}")
    
    # 1. Connect
    logging.info(f"Connecting to Milvus at {MILVUS_HOST}:{MILVUS_PORT}...")
    try:
        connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT)

    except Exception as e:
        logging.error(f"Failed to connect to Milvus: {e}")
        sys.exit(1)
    print("Connected to Milvus")
    # 2. Load Collection
    try:
        collection = Collection(COLLECTION_NAME)
        collection.load()
        logging.info(f"Collection '{COLLECTION_NAME}' loaded. Entities: {collection.num_entities}")
    except Exception as e:
        logging.error(f"Collection '{COLLECTION_NAME}' does not exist or failed to load! Please run dump_data.py first.")
        logging.error(f"Milvus error: {e}")
        sys.exit(1)
    
    # 3. Init Models
    try:
        logging.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME}...")
        embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME, trust_remote_code=True)
        
        logging.info("Loading reranking model...")
        reranker = RerankerWrapper(RERANKING_MODEL_NAME)
    except Exception as e:
        logging.error(f"Error loading models: {e}")
        sys.exit(1)
    
    # 4. Configure Retriever
    retriever = Retriever(max_workers=4)
    
    # Simplified Models Config
    models_config = {
        "TOP_K": 20,
        "QUERY": user_query,
        "source": "milvus_results",
        "connected_collection": {
            COLLECTION_NAME: collection
        },
        "embedding_model": embed_model,
        "reranking_model": reranker
        # No 'collections_in_use' needed for simplified logic
    }
    
    # 5. Execute Retrieval
    logging.info("Executing simplified retrieval pipeline...")
    try:
        results = retriever.run_query_manager(models_config)
    except Exception as e:
        logging.error(f"Error during retrieval: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
        
    final_chunks = results.get("milvus_results", [])
    
    print(f"\n=== Retrieved {len(final_chunks)} Chunks ===")
    for i, chunk in enumerate(final_chunks[:3]):
        print(f"[{i+1}] Score: {chunk.get('rerank_score', 0):.4f}")
        print(f"Content: {chunk['text'][:150]}...")
        print("-" * 50)
        
    # 6. Generate Answer
    logging.info("Generating answer with LLM...")
    answer = generate_answer_crew(user_query, final_chunks)
    
    print("\n" + "="*20 + " FAL Answer " + "="*20)
    print(answer)
    print("="*52 + "\n")

if __name__ == "__main__":
    main()
