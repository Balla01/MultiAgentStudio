import sys
import os
from unittest.mock import MagicMock, patch

# Add src to pythonpath
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from studio.data.ingestion import DataIngestionPipeline
from studio.core.database_manager import DatabaseManager

class MockProvider:
    def __init__(self):
        self.client = MagicMock()
        # Mock client behavior
        self.client.has_collection.return_value = False

def test_two_step_ingestion():
    print("Testing Two-Step Ingestion...")
    
    # 1. Setup Mocks
    db_manager = DatabaseManager()
    milvus_provider = MockProvider()
    db_manager.register_database("milvus", milvus_provider)
    
    # Mock MilvusClient and EmbeddingGenerator
    with patch("studio.data.ingestion.MilvusClient") as MockClient, \
         patch("studio.data.embedding.EmbeddingGenerator") as MockEmbedder:
        
        # Setup Client Mock
        local_client_instance = MagicMock()
        local_client_instance.has_collection.return_value = False
        MockClient.return_value = local_client_instance
        
        # Setup Embedder Mock
        embedder_instance = MagicMock()
        embedder_instance.generate.return_value = [[0.1] * 1024] # Dummy vector
        MockEmbedder.return_value = embedder_instance
        
        # 2. Run Pipeline
        pipeline = DataIngestionPipeline(db_manager)
        # Force inject the mocked embedder (since it's init in init)
        pipeline.embedder = embedder_instance

        # Create dummy file
        with open("verify_ingest.txt", "w") as f:
            f.write("Test content for ingestion.")
            
        try:
            print("Processing file...")
            pipeline.process_text_file(
                file_path="verify_ingest.txt", 
                target_dbs=["milvus"], 
                work_item_id="ticket_123"
            )
            
            # 3. Verify Local Dump
            # Check if MilvusClient was initialized with local path
            args, kwargs = MockClient.call_args
            # Handle if call_args is None (should not be if called)
            if not args:
                print("❌ MilvusClient was not instantiated.")
            else:
                uri_arg = kwargs.get('uri', args[0])
                print(f"MilvusClient init URI: {uri_arg}")
                assert "ticket_123.db" in str(uri_arg)
            
            # Check insert called on local client
            local_client_instance.insert.assert_called()
            print("✅ Local DB insert called.")
            
            # 4. Verify Server Dump
            # Server client is accessed via provider.client
            milvus_provider.client.insert.assert_called()
            # Check collection name passed to create/insert
            call_kwargs = milvus_provider.client.insert.call_args[1]
            assert call_kwargs['collection_name'] == "ticket_123"
            print("✅ Server DB insert called with correct collection name.")
            
        finally:
            if os.path.exists("verify_ingest.txt"):
                os.remove("verify_ingest.txt")
                
    print("✅ Two-Step Ingestion Logic Verified.")

if __name__ == "__main__":
    test_two_step_ingestion()
