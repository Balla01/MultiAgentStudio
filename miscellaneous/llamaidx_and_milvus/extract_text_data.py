from pymilvus import MilvusClient

def main():
    # Initialize Milvus Client
    client = MilvusClient("/home/ntlpt19/Downloads/TradeGpt_vec_db.db")
    
    print("Querying all text from collection...")
    
    # Query all entities. 
    # Using 'id >= 0' as a filter to retrieve all records (assuming integer IDs).
    # If your IDs are strings, you might need a different filter (e.g. 'id != ""').
    res = client.query(
        collection_name="demo_collection",
        filter="id >= 0", 
        output_fields=["text"],
        limit=16384  # Adjust limit if you have more documents
    )
    
    # Extract text results
    # client.query returns a list of dictionaries: [{'text': '...', 'id': ...}, ...]
    all_text = []
    for item in res:
        if 'text' in item:
            all_text.append(item['text'])
            
    # Concatenate all text
    final_text = "\n".join(all_text)
    
    # Write to File
    output_filename = "extracted_data.txt"
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write(final_text)
        
    print(f"Successfully extracted {len(all_text)} records to '{output_filename}'")

if __name__ == "__main__":
    main()
