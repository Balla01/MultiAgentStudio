from pymilvus import MilvusClient
import pandas as pd

# Connect to your Milvus Lite database
client = MilvusClient("/home/ntlpt19/Downloads/TradeGpt_vec_db.db")

print("✅ Connected to Milvus Lite database!")
print("="*80)

# List all collections
collections = client.list_collections()
print(f"\n📊 Total Collections: {len(collections)}")
print(f"Collections: {collections}")
print("="*80)

# Get detailed info for each collection
for col_name in collections:
    print(f"\n📁 Collection: {col_name}")
    print("-"*80)
    
    # Get collection stats
    stats = client.get_collection_stats(col_name)
    print(f"Total Records: {stats['row_count']}")
    
    # Describe collection (get schema)
    schema = client.describe_collection(col_name)
    print(f"\nSchema:")
    for field in schema['fields']:
        print(f"  • {field['name']} ({field['type']}) - Primary: {field.get('is_primary', False)}")
    
    print()

print("="*80)

for col_name in collections:
    print(f"\n{'='*80}")
    print(f"📊 Data from Collection: {col_name}")
    print('='*80)
    
    # Get total count
    stats = client.get_collection_stats(col_name)
    total_records = stats['row_count']
    print(f"Total Records: {total_records}")
    
    # Query first 10 records
    results = client.query(
        collection_name=col_name,
        filter="",  # Empty filter gets all
        limit=10,
        output_fields=["*"]
    )
    
    if results:
        # Display as DataFrame
        df = pd.DataFrame(results)
        print(f"\nFirst 10 records:")
        print(df.to_string(index=False))
    else:
        print("No data in this collection")
    
    print()