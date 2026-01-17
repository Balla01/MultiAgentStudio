import sys
import os
import shutil
from pathlib import Path

# Add src to pythonpath
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from studio.core import DatabaseManager
from studio.data import DataIngestionPipeline

class MockProvider:
    def dump_data(self, data, metadata=None):
        print(f"Mock dump_data called with {len(data) if isinstance(data, list) else 1} items")
        return True

def test_ingestion():
    print("Testing Ingestion Pipeline...")
    
    # Mock DB Manager
    db_manager = DatabaseManager()
    db_manager.register_database("milvus", MockProvider())
    db_manager.register_database("neo4j", MockProvider())
    db_manager.register_database("mysql", MockProvider())
    
    # Init Pipeline
    pipeline = DataIngestionPipeline(db_manager)
    print("✅ Pipeline initialized.")
    
    # Create dummy file
    dummy_file = Path("test_ingest.txt")
    dummy_file.write_text("Rule 1: test\nSource Document: doc1\n\nThis is a test content for chunking.", encoding="utf-8")
    
    try:
        # Test Process
        print("Processing file...")
        pipeline.process_text_file(str(dummy_file), ["milvus", "neo4j", "mysql"])
        print("✅ File processed.")
    finally:
        if dummy_file.exists():
            dummy_file.unlink()

if __name__ == "__main__":
    test_ingestion()
