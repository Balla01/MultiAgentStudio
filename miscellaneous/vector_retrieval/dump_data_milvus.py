import os
import sys
import logging
from pymilvus import MilvusClient, DataType, FieldSchema, CollectionSchema

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configuration

TARGET_URI = "http://localhost:19530"
DIMENSION = 1024
BATCH_SIZE = 50

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

def migrate(SOURCE_DB_FILE, COLLECTION_NAME):
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
    source_db_file = "/home/ntlpt19/personal_projects/MultiAgentStudio/data/vdbs/itf_testing.db"
    collection_name = "itf_testing"
    migrate(source_db_file, collection_name)
