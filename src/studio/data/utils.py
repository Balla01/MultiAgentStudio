import os
import sys
import json
import logging
import signal
import shutil
from typing import List, Dict, Any
from datetime import datetime
import numpy as np
from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer
from crewai import Agent, Task, Crew, Process, LLM

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
DB_FILE = "../data/vdbs"
# BACKUP_DIR = "../data/vdbs/backups"
# COLLECTION_NAME = "Trade_Finance_Article"
EMBEDDING_MODEL_NAME = "Alibaba-NLP/gte-large-en-v1.5"
EMBEDDING_DIM = 1024 
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Constants for CrewAI
MISTRAL_API_KEY = "0TD9nsBifR6Lkr1kOag9aikbCBImYfGg"
MISTRAL_MODEL = "mistral/mistral-large-latest"
CHECKPOINT_FILE = "dump_progress.json"


import os
import sys
import logging
from pymilvus import MilvusClient, DataType, FieldSchema, CollectionSchema

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configuration

TARGET_URI = "http://localhost:19530"
DIMENSION = 1024
embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME, trust_remote_code=True)


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


def connect_milvus_lite(db_path: str):
    """Connect to Milvus Lite (local .db file)"""
    print(f"Connecting to Milvus Lite database: {db_path}")
    
    # Create directory if it doesn't exist
    db_dir = os.path.dirname(db_path)
    if db_dir and not os.path.exists(db_dir):
        os.makedirs(db_dir)
    
    client = MilvusClient(db_path)
    print("✅ Connected to Milvus Lite!")
    return client

def create_collection_if_not_exists(client: MilvusClient, collection_name: str, dim: int, drop_if_exists: bool = True):
    """Create collection in Milvus Lite"""
    if client.has_collection(collection_name):
        if drop_if_exists:
            print(f"Collection '{collection_name}' already exists. Dropping it...")
            client.drop_collection(collection_name)
            print(f"Collection '{collection_name}' dropped.")
        else:
            print(f"Collection '{collection_name}' already exists. Resuming with existing collection.")
            return
    
    print(f"Creating collection '{collection_name}' with dimension {dim}...")
    client.create_collection(
        collection_name=collection_name,
        dimension=dim,
        metric_type="COSINE",
        auto_id=True
    )
    print(f"✅ Collection '{collection_name}' created!")

def generate_metadata(text_chunk: str, llm_instance: LLM, max_retries: int = 3) -> Dict[str, Any]:
    """
    Generates metadata for a given text chunk using CrewAI with Mistral.
    Includes retry logic for LLM failures.
    """
    for attempt in range(max_retries):
        try:
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
                description=f"Analyze the following text chunk and generate a JSON object containing keys: 'keywords' (list of strings), 'summary' (brief string), and 'entities' (list of strings). Just return the JSON string, nothing else.\n\nText Chunk:\n{text_chunk[:2000]}", 
                expected_output="A valid JSON string with keys: keywords, summary, entities.",
                agent=metadata_agent
            )

            crew = Crew(
                agents=[metadata_agent],
                tasks=[task],
                process=Process.sequential
            )

            result = crew.kickoff()
            result_str = str(result)
            
            # Basic cleanup to find JSON
            start = result_str.find('{')
            end = result_str.rfind('}') + 1
            if start != -1 and end != -1:
                json_str = result_str[start:end]
                return json.loads(json_str)
            else:
                logging.warning(f"Could not parse JSON from LLM response (attempt {attempt + 1}/{max_retries})")
                if attempt == max_retries - 1:
                    return {"keywords": [], "summary": "Failed to generate metadata", "entities": [], "error": "parsing_failed"}
                    
        except Exception as e:
            logging.error(f"Error generating metadata (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt == max_retries - 1:
                # Final fallback after all retries
                return {"keywords": [], "summary": "Error generating metadata", "entities": [], "error": str(e)}
            else:
                print(f"Retrying metadata generation... ({attempt + 1}/{max_retries})")
                import time
                time.sleep(2)  # Wait before retry
    
    return {"keywords": [], "summary": "Error generating metadata", "entities": [], "error": "max_retries_exceeded"}

def load_checkpoint(workitem: str) -> int:
    """Loads the last processed chunk index from the checkpoint file."""
    if os.path.exists(CHECKPOINT_FILE):
        try:
            with open(CHECKPOINT_FILE, 'r') as f:
                data = json.load(f)
                if data.get("workitem") == workitem:
                    return data.get("last_processed_index", -1)
        except Exception as e:
            logging.warning(f"Failed to load checkpoint: {e}")
    return -1

def save_checkpoint(source_file: str, workitem: str, index: int, total_chunks: int = None):
    """Saves the current chunk index to the checkpoint file with additional metadata."""
    try:
        checkpoint_data = {
            "source_file": source_file,
            "last_processed_index": index,
            "timestamp": datetime.now().isoformat(),
            "db_file": DB_FILE,
            "workitem": workitem,
        }
        if total_chunks:
            checkpoint_data["total_chunks"] = total_chunks
            checkpoint_data["progress_percentage"] = round((index + 1) / total_chunks * 100, 2)
        
        with open(CHECKPOINT_FILE, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
        
        logging.info(f"Checkpoint saved at index {index}")
    except Exception as e:
        logging.error(f"Failed to save checkpoint: {e}")

def get_collection_count(client: MilvusClient, collection_name: str) -> int:
    """Get the number of entities in a collection"""
    try:
        stats = client.get_collection_stats(collection_name)
        return stats.get("row_count", 0)
    except Exception as e:
        logging.warning(f"Could not get collection stats: {e}")
        return 0

def safe_insert_batch(client: MilvusClient, data_to_insert: List[Dict], 
                      source_file: str, workitem: str, current_index: int, total_chunks: int):
    """Safely insert batch with error handling and checkpoint saving"""                                                                                                                                  
    try:
        if not data_to_insert:
            return True
            
        client.insert(
            collection_name=workitem,
            data=data_to_insert
        )
        print(f"✅ Inserted batch of {len(data_to_insert)} records")
        client.flush(collection_name=workitem)
        stats = client.get_collection_stats(workitem)
        print("Entity count:", stats)
        # Save checkpoint after successful insert
        save_checkpoint(source_file, workitem, current_index, total_chunks)
    except Exception as e:
        logging.error(f"Failed to insert batch: {e}")
        exit('NOT DUMPED')


def main_dump(workitem, target_file):
    client = connect_milvus_lite(os.path.join(DB_FILE, workitem + '.db'))
    actual_dim = embed_model.get_sentence_embedding_dimension()
    print(f"Model loaded. Dimension: {actual_dim}")
    mistral_llm = LLM(
            model=MISTRAL_MODEL, 
            temperature=0.7, 
            api_key=MISTRAL_API_KEY
        )

    # 4. Check Checkpoint
    start_index = 0
    drop_collection = True
    
    last_index = load_checkpoint(workitem)
    if last_index >= 0:
        print(f"🔄 Found checkpoint! Resuming from chunk index {last_index + 1}...")
        start_index = last_index + 1
        drop_collection = False

    else:
        print("No valid checkpoint found. Starting fresh.")

    # 5. Get/Create Collection
    try:
        create_collection_if_not_exists(client, workitem, actual_dim, drop_if_exists=drop_collection)
    except Exception as e:
        print(f"Failed to create collection: {e}")
        return

    # 6. Read and Chunk File
    print(f"Reading file: {target_file}")
    try:
        with open(target_file, 'r', encoding='utf-8') as f:
            text_content = f.read()
    except Exception as e:
        print(f"Failed to read file: {e}")
        return

    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_text(text_content)
    total_chunks = len(chunks)
    print(f"Total chunks generated: {total_chunks}. Starting from: {start_index}")

    if start_index >= total_chunks:
        print("✅ Checkpoint indicates all chunks are already processed. Exiting.")
        return

    # 7. Process Chunks
    data_to_insert = []
    BATCH_SIZE = 5
    i = start_index  # Track current index for cleanup
    
    print("Processing chunks (Embedding + Metadata Generation)... This may take time.")
    print("Press Ctrl+C to safely stop and save progress.\n")
    
    # try:
    for i in range(start_index, total_chunks):
        chunk = chunks[i]
        print(f"Processing chunk {i+1}/{total_chunks} ({round((i+1)/total_chunks*100, 1)}%)...")
        
        # try:
        # Embed
        embedding = embed_model.encode(chunk).tolist()
        
        # Meta with retry logic
        meta = generate_metadata(chunk, mistral_llm, max_retries=3)
        
        # Check if metadata generation had errors
        if meta.get("error"):
            logging.warning(f"Metadata generation had issues for chunk {i+1}: {meta.get('error')}")
        
        # Prepare data for Milvus Lite format
        data_to_insert.append({
            "text": chunk,
            "vector": embedding,
            "metadata": meta
        })

        if len(data_to_insert) >= BATCH_SIZE:
            # Insert batch with error handling
            safe_insert_batch(client, data_to_insert, target_file, workitem, i, total_chunks)
            data_to_insert = []
            
    # Success! Get final count and backup
    total_count = get_collection_count(client, workitem) 
    print("\n" + "="*60)
    print("✅ DATA DUMP COMPLETE!")
    print("="*60)
    print(f"Total chunks processed: {total_chunks}")
    print(f"Total entities in collection: {total_count}")
    print(f"Database file: {DB_FILE}")







def connect_source(SOURCE_DB_FILE):
    """Connect to the local Milvus Lite database file."""
    if not os.path.exists(SOURCE_DB_FILE):
        logging.error(f"❌ Source database file not found: {SOURCE_DB_FILE}")
        sys.exit(1)
    
    try:
        logging.info(f"Connecting to Source (Milvus Lite): {SOURCE_DB_FILE}")
        client = MilvusClient(uri=SOURCE_DB_FILE)
        logging.info("✅ Connected to Source")
        return client
    except Exception as e:
        logging.error(f"❌ Failed to connect to source: {e}")
        sys.exit(1)

def connect_target():
    """Connect to the Milvus Server."""
    try:
        logging.info(f"Connecting to Target (Milvus Server): {TARGET_URI}")
        client = MilvusClient(uri=TARGET_URI)
        logging.info("✅ Connected to Target")
        return client
    except Exception as e:
        logging.error(f"❌ Failed to connect to target: {e}")
        logging.error("💡 checking if Milvus Docker container is running...")
        sys.exit(1)

def migrate(COLLECTION_NAME):
    SOURCE_DB_FILE = os.path.join(DB_FILE, COLLECTION_NAME + '.db')
    print(f"\n{'='*60}")
    print("MILVUS MIGRATION: Lite (.db) -> Server (localhost:19530)")
    print(f"{'='*60}\n")

    # 1. Connect clients
    source_client = connect_source(SOURCE_DB_FILE)
    target_client = connect_target()
    
    # 2. Verify Source Collection
    if not source_client.has_collection(COLLECTION_NAME):
        logging.error(f"❌ Collection '{COLLECTION_NAME}' not found in source database.")
        sys.exit(1)
        
    source_stats = source_client.get_collection_stats(COLLECTION_NAME)
    source_count = source_stats['row_count']
    logging.info(f"Found {source_count} entities in source collection.")
    
    if source_count == 0:
        logging.warning("⚠️ Source collection is empty. Nothing to migrate.")
        return

    # 3. Prepare Target Collection
    # To ensure clean migration and preserving IDs, we will Re-create the collection.
    if target_client.has_collection(COLLECTION_NAME):
        logging.info(f"Target collection '{COLLECTION_NAME}' exists. Dropping it for clean migration...")
        target_client.drop_collection(COLLECTION_NAME)
    
    logging.info(f"Creating collection '{COLLECTION_NAME}' on target...")
    
    # Define explicit schema to allow inserting custom IDs if needed, 
    # but based on previous context, we want to just dump data.
    # However, to preserve the exact IDs from the DB file, we must disable auto_id
    # and provide the IDs we read from the source.
    
    try:
        # Schema Definition
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=DIMENSION),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="metadata", dtype=DataType.JSON)
        ]
        schema = CollectionSchema(fields, description="Migrated Trade Finance Collection")
        
        index_params = target_client.prepare_index_params()
        index_params.add_index(
            field_name="vector", 
            index_type="HNSW", 
            metric_type="COSINE", 
            params={"M": 16, "efConstruction": 200}
        )

        target_client.create_collection(
            collection_name=COLLECTION_NAME,
            schema=schema,
            index_params=index_params
        )
        logging.info(f"✅ Collection created on target (auto_id=False).")
        
    except Exception as e:
        logging.error(f"❌ Failed to create collection on target: {e}")
        sys.exit(1)

    # 4. Fetch and Insert Data
    logging.info("Starting data transfer...")
    
    try:
        # Fetch all data from source
        # Milvus Lite limit is effectively memory-bound.
        # For larger datasets, we would need pagination.
        # Assuming dataset fits in memory for now based on previous interactions (< 2000 chunks).
        
        logging.info("Fetching data from source...")
        results = source_client.query(
            collection_name=COLLECTION_NAME,
            filter="", 
            output_fields=["id", "vector", "text", "metadata"],
            limit=source_count + 100 # Fetch all
        )
        
        logging.info(f"Fetched {len(results)} entities.")
        
        # Batch Insert
        data_to_insert = []
        total_migrated = 0
        BATCH_SIZE = 50

        for i, item in enumerate(results):
            # item dict keys match the schema: id, vector, text, metadata
            data_to_insert.append(item)
            
            if len(data_to_insert) >= BATCH_SIZE:
                target_client.insert(collection_name=COLLECTION_NAME, data=data_to_insert)
                total_migrated += len(data_to_insert)
                print(f"Migrated {total_migrated}/{source_count} entities...", end='\r')
                data_to_insert = []
        
        # Insert remaining
        if data_to_insert:
            target_client.insert(collection_name=COLLECTION_NAME, data=data_to_insert)
            total_migrated += len(data_to_insert)
        
        # EXPLICIT FLUSH TO ENSURE DATA VISIBILITY
        print("Flushing data to disk...")
        target_client.flush(COLLECTION_NAME)
            
        print(f"\n✅ Migration complete. Total inserted: {total_migrated}")
        
    except Exception as e:
        logging.error(f"❌ Error during data migration: {e}")
        sys.exit(1)

    # 5. Final Verification
    print(f"\n{'='*60}")
    print("VERIFICATION")
    print(f"{'='*60}")
    
    try:
        target_count = target_client.get_collection_stats(COLLECTION_NAME)['row_count']
        print(f"Source Count: {source_count}")
        print(f"Target Count: {target_count}")
        
        if target_count == source_count:
            print(f"\n✅ SUCCESS: All {target_count} entities migrated successfully.")
        else:
            print(f"\n⚠️  WARNING: Count mismatch! (Diff: {abs(target_count - source_count)})")
            
    except Exception as e:
        logging.error(f"Error verifying target: {e}")

if __name__ == "__main__":
    workitem = "itf_testing_123"
    target_file = "/home/ntlpt19/personal_projects/MultiAgentStudio/data/itf/extracted_data_temp.txt"
    main_dump(workitem, target_file)
    migrate(workitem)
