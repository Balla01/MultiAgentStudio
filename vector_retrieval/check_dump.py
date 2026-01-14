'''
from pymilvus import MilvusClient

# 1. Initialize the client
# Port 19530 is the default for Milvus Standalone
client = MilvusClient(
    uri="http://localhost:19530"
)

# 2. Check the connection by listing collections
collections = client.list_collections()
print(f"Connected! Collections in database: {collections}")

# 3. Optional: Get server version to verify
from pymilvus import utility
print(f"Milvus Server Version: {client.get_server_version()}")
exit('OGGGGGGGGGG')
'''

import json
import os
from pymilvus import connections, Collection, utility

# Constants
MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"
COLLECTION_NAME = "Trade_Finance_Article"
CHECKPOINT_FILE = "dump_progress.json"

def main():
    # 1. Check Checkpoint File
    print("=== Checkpoint Status ===")
    if os.path.exists(CHECKPOINT_FILE):
        try:
            with open(CHECKPOINT_FILE, 'r') as f:
                data = json.load(f)
                print(json.dumps(data, indent=2))
        except Exception as e:
            print(f"Error reading checkpoint file: {e}")
    else:
        print(f"No checkpoint file found at {CHECKPOINT_FILE}")
    print("\n")

    # 2. Connect to Milvus
    print("=== Milvus Collection Status ===")
    try:
        connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT)
        print("✅ Connected to Milvus")
    except Exception as e:
        print(f"❌ Failed to connect to Milvus: {e}")
        return

    if not utility.has_collection(COLLECTION_NAME):
        print(f"❌ Collection '{COLLECTION_NAME}' does not exist.")
        return

    collection = Collection(COLLECTION_NAME)
    collection.load() # Load into memory to query

    count = collection.num_entities
    print(f"Collection: {COLLECTION_NAME}")
    print(f"Total Entities: {count}")
    
    if count > 0:
        print("\n=== Sample Data (Last 3 entries) ===")
        # We can't easily get "last" by insertion order without a timestamp or incrementing ID known beforehand in a specific way,
        # but auto_id=True gives us IDs. We can try to query.
        
        # Limit to 3, retrieving metadata and text
        try:
            # Querying purely by limit
            res = collection.query(
                expr="", 
                output_fields=["id", "text", "metadata"], 
                limit=300
            )
            
            for i, item in enumerate(res):
                print(f"\n[Entry {i+1}]")
                print(f"ID: {item['id']}")
                meta = item['metadata']
                print(f"Metadata: {json.dumps(meta, indent=2)}")
                text_preview = item['text'][:200] + "..." if len(item['text']) > 200 else item['text']
                print(f"Text Preview: {text_preview}")
        except Exception as e:
            print(f"Error querying sample data: {e}")

    # Release memory
    collection.release()

if __name__ == "__main__":
    main()
