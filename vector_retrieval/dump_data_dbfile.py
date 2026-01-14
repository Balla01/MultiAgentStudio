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
DB_FILE = "../data/vdbs/trade_finance_vectors.db"
BACKUP_DIR = "../data/vdbs/backups"
COLLECTION_NAME = "Trade_Finance_Article"
EMBEDDING_MODEL_NAME = "Alibaba-NLP/gte-large-en-v1.5"
EMBEDDING_DIM = 1024 
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Constants for CrewAI
MISTRAL_API_KEY = "0TD9nsBifR6Lkr1kOag9aikbCBImYfGg"
MISTRAL_MODEL = "mistral/mistral-large-latest"
CHECKPOINT_FILE = "dump_progress.json"

# Global flag for graceful shutdown
SHUTDOWN_REQUESTED = False

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

def signal_handler(signum, frame):
    """Handle interrupt signals gracefully"""
    global SHUTDOWN_REQUESTED
    print("\n⚠️  Interrupt signal received. Preparing for safe shutdown...")
    SHUTDOWN_REQUESTED = True

def backup_db_file(reason: str = "manual"):
    """Create a backup of the .db file"""
    try:
        if not os.path.exists(DB_FILE):
            logging.warning(f"Database file {DB_FILE} does not exist. Skipping backup.")
            return None
            
        os.makedirs(BACKUP_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_filename = f"trade_finance_{reason}_{timestamp}.db"
        backup_path = os.path.join(BACKUP_DIR, backup_filename)
        
        print(f"📦 Creating backup: {backup_filename}")
        shutil.copy2(DB_FILE, backup_path)
        print(f"✅ Backup saved: {backup_path}")
        return backup_path
    except Exception as e:
        logging.error(f"Failed to backup database: {e}")
        return None

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

def save_checkpoint(source_file: str, index: int, total_chunks: int = None):
    """Saves the current chunk index to the checkpoint file with additional metadata."""
    try:
        checkpoint_data = {
            "source_file": source_file,
            "last_processed_index": index,
            "timestamp": datetime.now().isoformat(),
            "db_file": DB_FILE
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

def safe_insert_batch(client: MilvusClient, collection_name: str, data_to_insert: List[Dict], 
                      source_file: str, current_index: int, total_chunks: int) -> bool:
    """Safely insert batch with error handling and checkpoint saving"""
    try:
        if not data_to_insert:
            return True
            
        client.insert(
            collection_name=collection_name,
            data=data_to_insert
        )
        print(f"✅ Inserted batch of {len(data_to_insert)} records")
        
        # Save checkpoint after successful insert
        save_checkpoint(source_file, current_index, total_chunks)
        
        # Create periodic backup every 50 chunks
        if (current_index + 1) % 50 == 0:
            backup_db_file(reason=f"periodic_chunk_{current_index + 1}")
        
        return True
    except Exception as e:
        logging.error(f"Failed to insert batch: {e}")
        return False

def cleanup_and_exit(client: MilvusClient, data_to_insert: List[Dict], source_file: str, 
                     current_index: int, total_chunks: int, exit_code: int = 0, reason: str = "normal"):
    """Perform cleanup operations and exit safely"""
    print(f"\n{'='*60}")
    print(f"INITIATING SAFE SHUTDOWN: {reason}")
    print(f"{'='*60}")
    
    # Try to insert remaining data
    if data_to_insert:
        print(f"Attempting to save {len(data_to_insert)} pending records...")
        success = safe_insert_batch(client, COLLECTION_NAME, data_to_insert, 
                                    source_file, current_index, total_chunks)
        if success:
            print(f"✅ Saved {len(data_to_insert)} pending records")
        else:
            print(f"⚠️  Failed to save pending records. Last safe checkpoint: {current_index - len(data_to_insert)}")
            # Adjust checkpoint to last known good state
            save_checkpoint(source_file, current_index - len(data_to_insert), total_chunks)
    
    # Create final backup
    print("\n📦 Creating final backup before exit...")
    backup_path = backup_db_file(reason=reason)
    
    # Print summary
    print(f"\n{'='*60}")
    print("SHUTDOWN SUMMARY")
    print(f"{'='*60}")
    print(f"Last processed chunk: {current_index + 1}/{total_chunks}")
    print(f"Progress: {round((current_index + 1) / total_chunks * 100, 2)}%")
    print(f"Checkpoint file: {CHECKPOINT_FILE}")
    print(f"Database file: {DB_FILE}")
    if backup_path:
        print(f"Backup file: {backup_path}")
    print(f"\n💡 To resume: python {sys.argv[0]} {source_file}")
    print(f"{'='*60}\n")
    
    sys.exit(exit_code)

def main():
    global SHUTDOWN_REQUESTED
    
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # File Path Check
    if len(sys.argv) < 2:
        target_file = "../data/itf/extracted_data.txt"
        if not target_file:
            print("No file provided. Exiting.")
            return
    else:
        target_file = sys.argv[1]

    if not os.path.exists(target_file):
        print(f"File {target_file} does not exist.")
        return

    # 1. Connect to Milvus Lite
    try:
        client = connect_milvus_lite(DB_FILE)
    except Exception as e:
        print(f"Failed to connect to Milvus Lite: {e}")
        return

    # 2. Init Models
    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}...")
    try:
        embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME, trust_remote_code=True)
        actual_dim = embed_model.get_sentence_embedding_dimension()
        print(f"Model loaded. Dimension: {actual_dim}")
    except Exception as e:
        print(f"Error loading model {EMBEDDING_MODEL_NAME}: {e}")
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
        # Create a resume backup
        backup_db_file(reason="resume")
    else:
        print("No valid checkpoint found. Starting fresh.")

    # 5. Get/Create Collection
    try:
        create_collection_if_not_exists(client, COLLECTION_NAME, actual_dim, drop_if_exists=drop_collection)
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
    BATCH_SIZE = 10
    i = start_index  # Track current index for cleanup
    
    print("Processing chunks (Embedding + Metadata Generation)... This may take time.")
    print("Press Ctrl+C to safely stop and save progress.\n")
    
    try:
        for i in range(start_index, total_chunks):
            # Check for shutdown request
            if SHUTDOWN_REQUESTED:
                print("\n⚠️  Shutdown requested. Saving progress...")
                cleanup_and_exit(client, data_to_insert, target_file, i - 1, 
                               total_chunks, exit_code=0, reason="user_interrupt")
            
            chunk = chunks[i]
            print(f"Processing chunk {i+1}/{total_chunks} ({round((i+1)/total_chunks*100, 1)}%)...")
            
            try:
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
                    success = safe_insert_batch(client, COLLECTION_NAME, data_to_insert, 
                                               target_file, i, total_chunks)
                    
                    if not success:
                        print("⚠️  Batch insert failed. Stopping...")
                        cleanup_and_exit(client, [], target_file, i - len(data_to_insert), 
                                       total_chunks, exit_code=1, reason="insert_failure")
                    
                    data_to_insert = []
                    
            except KeyboardInterrupt:
                # This will be caught by the outer try-except
                raise
                
            except Exception as e:
                logging.error(f"Error processing chunk {i+1}: {e}")
                print(f"⚠️  Error on chunk {i+1}. Attempting to save progress...")
                
                # Try to save what we have
                if data_to_insert:
                    success = safe_insert_batch(client, COLLECTION_NAME, data_to_insert, 
                                               target_file, i - 1, total_chunks)
                    if success:
                        cleanup_and_exit(client, [], target_file, i - 1, 
                                       total_chunks, exit_code=1, reason="processing_error")
                    else:
                        cleanup_and_exit(client, [], target_file, i - len(data_to_insert) - 1, 
                                       total_chunks, exit_code=1, reason="processing_error")
                else:
                    cleanup_and_exit(client, [], target_file, i - 1, 
                                   total_chunks, exit_code=1, reason="processing_error")

        # Insert remaining
        if data_to_insert:
            success = safe_insert_batch(client, COLLECTION_NAME, data_to_insert, 
                                       target_file, total_chunks - 1, total_chunks)
            if not success:
                print("⚠️  Final batch insert failed.")
                cleanup_and_exit(client, [], target_file, i - len(data_to_insert), 
                               total_chunks, exit_code=1, reason="final_insert_failure")

        # Success! Get final count and backup
        total_count = get_collection_count(client, COLLECTION_NAME)
        print("\n" + "="*60)
        print("✅ DATA DUMP COMPLETE!")
        print("="*60)
        print(f"Total chunks processed: {total_chunks}")
        print(f"Total entities in collection: {total_count}")
        print(f"Database file: {DB_FILE}")
        
        # Create final success backup
        print("\n📦 Creating final backup...")
        backup_path = backup_db_file(reason="complete")
        if backup_path:
            print(f"✅ Final backup: {backup_path}")
        
        # Clean up checkpoint file on success
        if os.path.exists(CHECKPOINT_FILE):
            os.remove(CHECKPOINT_FILE)
            print(f"✅ Checkpoint file removed (processing complete)")
        
        print("="*60 + "\n")
        
    except KeyboardInterrupt:
        print("\n⚠️  Keyboard interrupt detected...")
        cleanup_and_exit(client, data_to_insert, target_file, i - 1, 
                        total_chunks, exit_code=0, reason="keyboard_interrupt")
    
    except Exception as e:
        logging.error(f"Unexpected error in main loop: {e}")
        print(f"\n⚠️  Unexpected error: {e}")
        cleanup_and_exit(client, data_to_insert, target_file, i - 1, 
                        total_chunks, exit_code=1, reason="unexpected_error")

if __name__ == "__main__":
    main()