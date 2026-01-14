import os
import sys
import json
import logging
import threading
from typing import List, Dict, Any
import numpy as np
from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)
from sentence_transformers import SentenceTransformer
from crewai import Agent, Task, Crew, Process, LLM

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"
COLLECTION_NAME = "Trade_Finance_Article"  # Default collection
EMBEDDING_MODEL_NAME = "Alibaba-NLP/gte-large-en-v1.5"
# gte-large-en-v1.5 typically has 1024 dimensions.
# We will verify or use a standard if loading fails, but let's assume 1024 for this model.
EMBEDDING_DIM = 1024 
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Constants for CrewAI
MISTRAL_API_KEY = "0TD9nsBifR6Lkr1kOag9aikbCBImYfGg"
MISTRAL_MODEL = "mistral/mistral-large-latest"
CHECKPOINT_FILE = "dump_progress.json"

class RecursiveCharacterTextSplitter:
    """
    A simple implementation of recursive character text splitting.
    """
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200, separators: List[str] = ["\n\n", "\n", " ", ""]):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators

    def split_text(self, text: str) -> List[str]:
        final_chunks = []
        if self._length_function(text) <= self.chunk_size:
            return [text]
        
        # Try separators
        separator = self.separators[-1]
        for sep in self.separators:
            if sep in text:
                separator = sep
                break
        
        splits = text.split(separator) if separator else list(text)
        good_splits = []
        
        for split in splits:
            if self._length_function(split) < self.chunk_size:
                good_splits.append(split)
            else:
                if good_splits:
                    self._merge_splits(good_splits, separator, final_chunks)
                    good_splits = []
                final_chunks.extend(self.split_text(split))
        
        if good_splits:
            self._merge_splits(good_splits, separator, final_chunks)
            
        return final_chunks

    def _length_function(self, text: str) -> int:
        return len(text)

    def _merge_splits(self, splits: List[str], separator: str, final_chunks: List[str]):
        current_chunk = []
        current_length = 0
        
        for split in splits:
            split_len = self._length_function(split)
            if current_length + split_len + (len(separator) if current_length > 0 else 0) > self.chunk_size:
                if current_chunk:
                    doc = separator.join(current_chunk)
                    final_chunks.append(doc)
                    
                    # Handle overlap
                    while current_length > self.chunk_overlap:
                        current_length -= self._length_function(current_chunk[0]) + len(separator)
                        current_chunk.pop(0)
                        
            current_chunk.append(split)
            current_length += split_len + (len(separator) if current_length > 0 else 0)
            
        if current_chunk:
            final_chunks.append(separator.join(current_chunk))

def connect_milvus():
    print(f"Connecting to Milvus at {MILVUS_HOST}:{MILVUS_PORT}...")
    connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT)
    print("✅ Connected to Milvus!")

def create_collection_if_not_exists(collection_name: str, dim: int, drop_if_exists: bool = True):
    if utility.has_collection(collection_name):
        if drop_if_exists:
            print(f"Collection '{collection_name}' already exists. Dropping it...")
            utility.drop_collection(collection_name)
            print(f"Collection '{collection_name}' dropped.")
        else:
            print(f"Collection '{collection_name}' already exists. Resuming with existing collection.")
            return Collection(collection_name)
    
    print(f"Creating collection '{collection_name}'...")
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
        FieldSchema(name="metadata", dtype=DataType.JSON)
    ]
    schema = CollectionSchema(fields, description="Trade Finanace DSS Collection")
    collection = Collection(name=collection_name, schema=schema)
    
    # Create Index
    index_params = {
        "metric_type": "COSINE",
        "index_type": "HNSW",
        "params": {"M": 16, "efConstruction": 200}
    }
    collection.create_index(field_name="embedding", index_params=index_params)
    print(f"✅ Collection '{collection_name}' and index created!")
    return collection

def generate_metadata(text_chunk: str, llm_instance: LLM) -> Dict[str, Any]:
    """
    Generates metadata for a given text chunk using CrewAI with Mistral.
    """
    # Create an agent
    metadata_agent = Agent(
        role='Metadata Specialist',
        goal='Generate structured metadata for document chunks',
        backstory='You are an expert in information retrieval and metadata tagging. You analyze text to extract key information.',
        llm=llm_instance,
        verbose=False
    )

    # Define the task
    task = Task(
        description=f"Analyze the following text chunk and generate a JSON object containing keys: 'keywords' (list of strings), 'summary' (brief string), and 'entities' (list of strings). Just return the JSON string, nothing else.\n\nText Chunk:\n{text_chunk[:2000]}", # Limit context if needed
        expected_output="A valid JSON string with keys: keywords, summary, entities.",
        agent=metadata_agent
    )

    crew = Crew(
        agents=[metadata_agent],
        tasks=[task],
        process=Process.sequential
    )

    try:
        result = crew.kickoff()
        # Parse result to JSON
        # The output might be a raw string, we try to parse it.
        # CrewAI returns a TaskOutput, convert to string
        result_str = str(result)
        
        # Basic cleanup to find JSON
        start = result_str.find('{')
        end = result_str.rfind('}') + 1
        if start != -1 and end != -1:
            json_str = result_str[start:end]
            return json.loads(json_str)
        else:
            return {"error": "Could not parse JSON", "raw": result_str}
    except Exception as e:
        logging.error(f"Error generating metadata: {e}")
        return {"keywords": [], "summary": "Error generating metadata", "entities": []}

def load_checkpoint(source_file: str) -> int:
    """Loads the last processed chunk index from the checkpoint file."""
    if os.path.exists(CHECKPOINT_FILE):
        try:
            with open(CHECKPOINT_FILE, 'r') as f:
                data = json.load(f)
                if data.get("source_file") == source_file:
                    return data.get("last_processed_index", -1)
        except Exception as e:
            logging.warning(f"Failed to load checkpoint: {e}")
    return -1

def save_checkpoint(source_file: str, index: int):
    """Saves the current chunk index to the checkpoint file."""
    try:
        with open(CHECKPOINT_FILE, 'w') as f:
            json.dump({
                "source_file": source_file,
                "last_processed_index": index
            }, f)
    except Exception as e:
        logging.error(f"Failed to save checkpoint: {e}")

def main():
    # File Path Check
    if len(sys.argv) < 2:
        # Default fallback or user prompt
        target_file = "/home/ntlpt19/personal_projects/CompanyResearch/llamaidx_and_milvus/extracted_data.txt"
        if not target_file:
            print("No file provided. Exiting.")
            return
    else:
        target_file = sys.argv[1]

    if not os.path.exists(target_file):
        print(f"File {target_file} does not exist.")
        return

    # 1. Connect to Milvus
    connect_milvus()

    # 2. Init Models
    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}...")
    try:
        embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME, trust_remote_code=True)
        # Check dim
        actual_dim = embed_model.get_sentence_embedding_dimension()
        print(f"Model loaded. Dimension: {actual_dim}")
    except Exception as e:
        print(f"Error loading model {EMBEDDING_MODEL_NAME}: {e}")
        print("Falling back to local default or exiting...")
        return

    # 3. Init LLM
    print("Initializing Mistral LLM via CrewAI...")
    try:
        mistral_llm = LLM(
            model=MISTRAL_MODEL, 
            temperature=0.7, 
            api_key=MISTRAL_API_KEY
        )
    except Exception as e:
        print(f"Error initializing LLM: {e}")
        return

    # 4. Check Checkpoint
    start_index = 0
    drop_collection = True
    
    last_index = load_checkpoint(target_file)
    if last_index >= 0:
        print(f"🔄 Found checkpoint! Resuming from chunk index {last_index + 1}...")
        start_index = last_index + 1
        drop_collection = False
    else:
        print("No valid checkpoint found. Starting fresh.")

    # 5. Get/Create Collection
    collection = create_collection_if_not_exists(COLLECTION_NAME, actual_dim, drop_if_exists=drop_collection)

    # 6. Read and Chunk File
    print(f"Reading file: {target_file}")
    with open(target_file, 'r', encoding='utf-8') as f:
        text_content = f.read()

    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_text(text_content)
    print(f"Total chunks generated: {len(chunks)}. Starting from: {start_index}")

    if start_index >= len(chunks):
        print("✅ Checkpoint indicates all chunks are already processed. Exiting.")
        return

    # 7. Process Chunks
    data_to_insert = {
        "text": [],
        "embedding": [],
        "metadata": []
    }

    print("Processing chunks (Embedding + Metadata Generation)... This may take time.")
    
    BATCH_SIZE = 10
    
    try:
        for i in range(start_index, len(chunks)):
            chunk = chunks[i]
            print(f"Processing chunk {i+1}/{len(chunks)}...")
            
            try:
                # Embed
                embedding = embed_model.encode(chunk).tolist()
                
                # Meta
                meta = generate_metadata(chunk, mistral_llm)
                
                data_to_insert["text"].append(chunk)
                data_to_insert["embedding"].append(embedding)
                data_to_insert["metadata"].append(meta)

                if len(data_to_insert["text"]) >= BATCH_SIZE:
                     collection.insert([
                         data_to_insert["text"],
                         data_to_insert["embedding"],
                         data_to_insert["metadata"]
                     ])
                     print(f"Inserted batch of {len(data_to_insert['text'])}")
                     
                     # Simple logic: save checkpoint after successful batch insert
                     # Ideally we track exactly which batch, but here i is the last processed index
                     save_checkpoint(target_file, i)
                     
                     data_to_insert = {"text": [], "embedding": [], "metadata": []}
            except Exception as e:
                logging.error(f"Error processing chunk {i+1}: {e}")
                print("⚠️ Stopping due to error. Saving progress...")
                
                # Try to flush whatever we have in the buffer so we don't lose it
                if data_to_insert["text"]:
                    try:
                        collection.insert([
                            data_to_insert["text"],
                            data_to_insert["embedding"],
                            data_to_insert["metadata"]
                        ])
                        # IMPORTANT: Flush to persist data!
                        collection.flush()
                        print(f"Inserted buffer of {len(data_to_insert['text'])} items before error.")
                        # If insert succeeds, our last successful index is i - 1
                        save_checkpoint(target_file, i - 1)
                    except Exception as insert_error:
                        logging.error(f"Failed to flush buffer on error: {insert_error}")
                        # If flush fails, we can't count those as done.
                        last_safe_index = i - len(data_to_insert["text"]) - 1
                        save_checkpoint(target_file, last_safe_index)
                else:
                    # No buffer, just save i-1
                    save_checkpoint(target_file, i - 1)
                
                sys.exit(1)

        # Insert remaining
        if data_to_insert["text"]:
            collection.insert([
                data_to_insert["text"],
                data_to_insert["embedding"],
                data_to_insert["metadata"]
            ])
            print(f"Inserted final batch of {len(data_to_insert['text'])}")
            save_checkpoint(target_file, len(chunks) - 1)

        # Flush and Index
        collection.flush()
        print("✅ Data dump complete!")
        print(f"Total entities in collection: {collection.num_entities}")
        
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user. Saving progress...")
        
        # Flush buffer
        if data_to_insert["text"]:
            try:
                collection.insert([
                    data_to_insert["text"],
                    data_to_insert["embedding"],
                    data_to_insert["metadata"]
                ])
                # IMPORTANT: Flush to persist data!
                collection.flush()
                print(f"Inserted buffer of {len(data_to_insert['text'])} items before exiting.")
                # i is the item being processed when interrupted.
                # processed items are 0..i-1 (which are in buffer).
                # So if buffer insert succeeds, last done is i-1.
                save_checkpoint(target_file, i - 1)
                print(f"Checkpoint saved at index {i - 1}.")
            except Exception as e:
                print(f"Error flushing buffer on interrupt: {e}")
                # Fallback
                last_safe_index = i - len(data_to_insert["text"]) - 1
                save_checkpoint(target_file, last_safe_index)
                print(f"Checkpoint saved at index {last_safe_index} (buffer flushed failed).")
        else:
            # Buffer empty, so i-1 is safe? 
            # If i was just started, i-1 is the last completed.
            save_checkpoint(target_file, i - 1)
            print(f"Checkpoint saved at index {i - 1}.")
            
        sys.exit(0)

if __name__ == "__main__":
    main()
