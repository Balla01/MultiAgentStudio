from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
from pymilvus import model
from llama_index.core.node_parser import SemanticDoubleMergingSplitterNodeParser, LanguageConfig
from llama_index.core import SimpleDirectoryReader

# Configuration
MILVUS_HOST = 'localhost'
MILVUS_PORT = '19530'
COLLECTION_NAME = 'my_text_collection'

# Connect to Milvus
connections.connect(
    alias="default", 
    host=MILVUS_HOST,
    port=MILVUS_PORT,
    timeout=30
)
print("✅ Connected to Milvus!")

# Initialize embedding model
sentence_transformer_ef = model.dense.SentenceTransformerEmbeddingFunction(
    model_name='all-MiniLM-L6-v2',
    device='cpu'
)

def chunk_text_from_file(file_path):
    """Load and chunk text file using semantic splitter"""
    documents = SimpleDirectoryReader(input_files=[file_path]).load_data()
    
    config = LanguageConfig(language="english", spacy_model="en_core_web_md")
    splitter = SemanticDoubleMergingSplitterNodeParser(
        language_config=config,
        initial_threshold=0.6,
        appending_threshold=0.7,
        merging_threshold=0.7,
    )
    
    nodes = splitter.get_nodes_from_documents(documents)
    return nodes

def create_and_load_collection(file_path, collection_name=COLLECTION_NAME):
    """Create collection and load data from text file"""
    
    # Drop collection if exists
    if utility.has_collection(collection_name):
        print(f"⚠️  Collection '{collection_name}' exists. Dropping it...")
        utility.drop_collection(collection_name)
    
    # Define schema
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=384),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=256)
    ]
    
    schema = CollectionSchema(fields, description="Text document collection")
    collection = Collection(name=collection_name, schema=schema)
    print(f"✅ Collection '{collection_name}' created!")
    
    # Load and chunk the text file
    print(f"📄 Loading file: {file_path}")
    nodes = chunk_text_from_file(file_path)
    docs = [node.get_content() for node in nodes]
    
    print(f"\n📊 Chunking Summary:")
    print(f"Total chunks: {len(docs)}")
    for i, doc in enumerate(docs[:3], 1):  # Show first 3 chunks
        print(f"\nChunk {i}:")
        print(f"  Length: {len(doc)} characters")
        print(f"  Preview: {doc[:100]}...")
    
    # Generate embeddings
    print(f"\n🔄 Generating embeddings...")
    vectors = sentence_transformer_ef.encode_documents(docs)
    print(f"✅ Generated {len(vectors)} embeddings (dim: {vectors[0].shape[0]})")
    
    # Prepare data for insertion
    data = [
        {
            "id": i, 
            "vector": vectors[i].tolist(), 
            "text": docs[i], 
            "source": file_path.split('/')[-1]
        }
        for i in range(len(vectors))
    ]
    
    # Insert data
    print(f"\n💾 Inserting {len(data)} records into Milvus...")
    collection.insert(data)
    collection.flush()
    print(f"✅ Data inserted successfully!")
    
    # Create index
    print(f"\n🔍 Creating index...")
    index_params = {
        "index_type": "HNSW",
        "metric_type": "L2",
        "params": {
            "M": 16,              # Connectivity: number of edges per node (8-64)
            "efConstruction": 200 # Search scope during index building (higher = better quality)
        }
    }
    collection.create_index(field_name="vector", index_params=index_params)
    print(f"✅ Index created!")
    
    return collection

def search_text(query_text, collection_name=COLLECTION_NAME, top_k=3):
    """Search for similar text chunks"""
    collection = Collection(collection_name)
    collection.load()
    
    # Encode query
    query_vector = sentence_transformer_ef.encode_queries([query_text])[0]
    
    # Search
    search_params = {
        "metric_type": "L2",
        "params": {"nprobe": 10},
    }
    
    results = collection.search(
        data=[query_vector.tolist()],
        anns_field="vector",
        param=search_params,
        limit=top_k,
        output_fields=["text", "source"]
    )
    
    # Format results
    similar_docs = []
    for hits in results:
        for hit in hits:
            similar_docs.append({
                "id": hit.id,
                "distance": hit.distance,
                "text": hit.entity.get("text"),
                "source": hit.entity.get("source")
            })
    
    return similar_docs

def display_search_results(query, results):
    """Display search results in a nice format"""
    print(f"\n{'='*80}")
    print(f"🔍 Query: {query}")
    print(f"{'='*80}\n")
    
    for i, result in enumerate(results, 1):
        print(f"Result {i}:")
        print(f"  📊 Distance: {result['distance']:.4f}")
        print(f"  📁 Source: {result['source']}")
        print(f"  📝 Text: {result['text'][:200]}...")
        print(f"  {'-'*76}\n")

# ============================================
# MAIN EXECUTION
# ============================================
if __name__ == '__main__':
    
    # ===== STEP 1: Load your .txt file =====
    txt_file_path = '/home/ntlpt19/Documents/itf_rules.txt'
    
    # Create collection and load data
    # create_and_load_collection(txt_file_path) ############
    
    print(f"\n{'='*80}")
    print(f"✅ SUCCESS! Data loaded into Milvus collection: '{COLLECTION_NAME}'")
    print(f"{'='*80}\n")
    
    # ===== STEP 2: Test search =====
    query = "Explain Rule 'BILL AMOUNT SHOULD LESS THAN OTHER DOCUMENTS'?"  # CHANGE THIS TO YOUR QUERY
    results = search_text(query, top_k=3)
    display_search_results(query, results)
    
    # ===== STEP 3: View collection stats =====
    collection = Collection(COLLECTION_NAME)
    collection.load()
    print(f"\n📊 Collection Stats:")
    print(f"  Total records: {collection.num_entities}")
    print(f"  Collection name: {COLLECTION_NAME}")