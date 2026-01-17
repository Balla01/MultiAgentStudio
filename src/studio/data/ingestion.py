import os
import json
import logging
from typing import List, Dict, Any
from pathlib import Path
import re

try:
    from pymilvus import MilvusClient, DataType, FieldSchema, CollectionSchema
except ImportError:
    logging.warning("pymilvus not found. Milvus ingestion will fail.")

from studio.core.database_manager import DatabaseManager
from studio.data.chunking import TextChunker
from studio.data.embedding import EmbeddingGenerator
from studio.data.utils import main_dump, migrate

logger = logging.getLogger(__name__)

class DataIngestionPipeline:
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.chunker = TextChunker() # Uses new 1000/200 defaults
        self.embedder = EmbeddingGenerator()
        
        # Ensure vdbs directory exists
        self.vdbs_dir = Path("data/vdbs")
        self.vdbs_dir.mkdir(parents=True, exist_ok=True)

    def process_text_file(self, file_path: str, target_dbs: List[str], work_item_id: str = None):
        """
        Process a text file and ingest into selected databases.
        
        :param work_item_id: Optional ID to be used as Collection Name and DB filename for Milvus.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        try:
            content = path.read_text(encoding='utf-8')
        except Exception as e:
            logger.error(f"Failed to read file {file_path}: {e}")
            return

        # 1. Chunking for Vectors
        chunks = self.chunker.split_text(content)
        
        # 2. Vector Store Ingestion (Milvus)
        # Modified Workflow: Local .db -> Server Migration
        if 'milvus' in target_dbs:
            if not work_item_id:
                # Fallback if no ID provided
                work_item_id = "Default_Collection"
                logger.warning(f"No Work Item ID provided. Using default: {work_item_id}")
            
            self._ingest_to_milvus_two_step(chunks, file_path, work_item_id)

        # 3. Graph Ingestion (Neo4j)
        if 'neo4j' in target_dbs:
            # Check if it looks like a rules file
            if "Rule" in content and "Source Document" in content:
                self._ingest_rules_to_neo4j(content)
            else:
                logger.warning(f"File {file_path} does not strictly match Rules format, skipping Neo4j ingestion for now.")

        # 4. Structured Store Ingestion (MySQL/SQLite)
        if 'mysql' in target_dbs:
            import pandas as pd
            import datetime
            
            df = pd.DataFrame([{
                "file_name": path.name,
                "work_item_id": work_item_id,
                "ingestion_time": str(datetime.datetime.now()),
                "status": "SUCCESS",
                "chunks": len(chunks)
            }])
            
            provider = self.db_manager.get_provider('mysql')
            if provider:
                provider.dump_data(df, metadata={"table_name": "ingestion_log"})

    def _ingest_to_milvus_two_step(self, chunks: List[str], source: str, work_item_id: str):
        """
        Step 1: Dump to local .db file (Milvus Lite).
        Step 2: Dump to Milvus Server.
        Collection Name = work_item_id.
        """

        # workitem = "itf_testing_123"
        # target_file = "/home/ntlpt19/personal_projects/MultiAgentStudio/data/itf/extracted_data_temp.txt"
        main_dump(work_item_id, source)
        migrate(work_item_id)


        # local_db_path = self.vdbs_dir / f"{work_item_id}.db"
        # collection_name = work_item_id
        
        # logger.info(f"Step 1: Ingesting into local DB: {local_db_path}")
        
        # # Generate Embeddings
        # logger.info(f"Generating embeddings for {len(chunks)} chunks...")
        # vectors = self.embedder.generate(chunks) # Returns list of lists
        
        # # Prepare Data
        # data = []
        # for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
        #     data.append({
        #         "text": chunk,
        #         "vector": vector,
        #         "metadata": {"source": source, "chunk_id": i}
        #     })
            
        # if not data:
        #     logger.warning("No data to ingest.")
        #     return

        # # --- Step 1: Local Dump ---
        # try:
        #     local_client = MilvusClient(uri=str(local_db_path))
            
        #     # Create/Recreate Collection
        #     if local_client.has_collection(collection_name):
        #         local_client.drop_collection(collection_name)
            
        #     local_client.create_collection(
        #         collection_name=collection_name,
        #         dimension=len(vectors[0]), # Auto-detect dim
        #         metric_type="COSINE",
        #         auto_id=True # Local DB uses auto_id
        #     )
            
        #     # Insert
        #     local_client.insert(collection_name=collection_name, data=data)
        #     logger.info(f"✅ Saved {len(data)} chunks to local DB {local_db_path}")
            
        # except Exception as e:
        #     logger.error(f"Failed to ingest into local DB: {e}")
        #     return # Stop if local dump fails

        # # --- Step 2: Server Dump ---
        # logger.info("Step 2: Migrating to Milvus Server...")
        # provider = self.db_manager.get_provider('milvus')
        # if not provider:
        #     logger.error("Milvus Server provider not configured. Skipping server dump.")
        #     return

        # # We can reuse the 'data' list we already have in memory!
        # # This saves us reading back from the local DB.
        # # But per requirements: "once this is done then automatically need to dump same into the milvus data base"
        # # We effectively just did the local part. Now we do the server part.
        
        # # Note: The Server Provider's dump_data usually handles collection creation.
        # # We need to ensure it uses the correct collection name (work_item_id).
        # # The current MilvusProvider might adhere to a config-based collection name.
        # # We might need to override it or use the provider's client directly if exposed,
        # # OR pass the collection name to dump_data if supported. 
        # # Checking logic: MilvusProvider.dump_data usually inserts into self.collection_name.
        # # We should modify the call to support dynamic collection name, or instantiate a transient one.
        
        # # Let's try to override the collection name temporarily or pass it if extended.
        # # Since I can't easily change the Provider interface right now without checking,
        # # I will check if provider allows switching collection.
        # # If not, I'll use provider.client (MilvusClient) directly if accessible.
        
        # try:
        #     server_client = provider.client # accessing internal client
            
        #     if server_client.has_collection(collection_name):
        #         logger.info(f"Collection {collection_name} exists on server. Appending...")
        #     else:
        #         logger.info(f"Creating collection {collection_name} on server...")
        #         server_client.create_collection(
        #              collection_name=collection_name,
        #              dimension=len(vectors[0]),
        #              metric_type="COSINE",
        #              auto_id=True
        #         )
            
        #     server_client.insert(collection_name=collection_name, data=data)
        #     logger.info(f"✅ Migrated {len(data)} chunks to Milvus Server (Collection: {collection_name})")
            
        # except Exception as e:
        #     logger.error(f"Failed to ingest into Milvus Server: {e}")

    def _ingest_rules_to_neo4j(self, content: str):
        provider = self.db_manager.get_provider('neo4j')
        if not provider:
            logger.error("Neo4j provider not found")
            return

        rules = self._parse_rules(content)
        logger.info(f"Parsed {len(rules)} rules for Neo4j ingestion.")
        provider.dump_data(rules)

    def _parse_rules(self, text: str) -> List[Dict[str, Any]]:
        # Logic adapted from neo4j_with_vector/a.py
        # Simplified for brevity, but should match core regexes
        
        def split_rule_blocks(text: str) -> List[str]:
            parts = re.split(r'(?m)^(Rule\s+\d+:\s+)', text)
            blocks = []
            if len(parts) <= 1: return [text]
            it = iter(parts)
            next(it) # skip preamble
            for sep in it:
                body = next(it, "")
                blocks.append(sep + body)
            return [b.strip() for b in blocks if b.strip()]

        def simple_extract(pattern: str, text: str):
            m = re.search(pattern, text, flags=re.IGNORECASE)
            return m.group(1).strip() if m else None

        blocks = split_rule_blocks(text)
        rules = []
        for block in blocks:
            rule_id = simple_extract(r'Rule\s+\d+:\s*([^\n\r]+)', block)
            if not rule_id: continue
            
            rules.append({
                "id": rule_id,
                "name": simple_extract(r'Rule Name:\s*(.*)', block),
                "group": simple_extract(r'Rule Group:\s*(.*)', block),
                "description": simple_extract(r'Rule Information:\s*(.*)', block),
                # Add more fields as needed
            })
        return rules




# ==================================================
