from pymilvus import connections, utility, Collection
import pandas as pd

# Connect
connections.connect(alias="default", host="localhost", port="19530")
print("✅ Connected to Milvus!")

# List all collections
collections = utility.list_collections()
print(f"\n📊 Total Collections: {len(collections)}")
print("="*80)

if len(collections) == 0:
    print("No collections found. Your Milvus database is empty.")
else:
    # Get details for each collection
    collection_data = []
    
    for col_name in collections:
        col = Collection(col_name)
        col.load()  # Load collection to get stats
        
        # Get collection info
        num_entities = col.num_entities
        schema = col.schema
        
        collection_data.append({
            'Collection Name': col_name,
            'Total Records': num_entities,
            'Fields': len(schema.fields),
            'Description': schema.description or 'N/A'
        })
    
    # Display as table
    df = pd.DataFrame(collection_data)
    print("\n" + df.to_string(index=False))
    
    print("\n" + "="*80)
    
    # Show detailed info for each collection
    for col_name in collections:
        print(f"\n📁 Collection: {col_name}")
        print("-"*80)
        
        col = Collection(col_name)
        schema = col.schema
        
        print(f"Description: {schema.description or 'N/A'}")
        print(f"Total Records: {col.num_entities}")
        print(f"\nFields:")
        
        for field in schema.fields:
            print(f"  • {field.name} ({field.dtype}) - Primary: {field.is_primary}, AutoID: {field.auto_id}")
        
        # Show indexes
        print(f"\nIndexes:")
        for field in schema.fields:
            if field.dtype in [101, 102, 103]:  # Vector types
                try:
                    index_info = col.index()
                    print(f"  • {field.name}: {index_info}")
                except:
                    print(f"  • {field.name}: No index")

############################################################
############################################################
############################################################
print('################################################')
print('################################################')
for col_name in collections:
    print(f"\n{'='*80}")
    print(f"📊 Data from Collection: {col_name}")
    print('='*80)
    
    col = Collection(col_name)
    col.load()
    
    # Query first 10 records
    results = col.query(
        expr="",  # Empty expr gets all
        limit=10,
        output_fields=["*"]
    )
    
    if results:
        # Display as DataFrame
        df = pd.DataFrame(results)
        print(df.to_string(index=False))
    else:
        print("No data in this collection")