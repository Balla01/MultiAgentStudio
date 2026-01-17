import sys
import os

# Add src to pythonpath
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from studio.core import DatabaseManager, MilvusProvider, Neo4jProvider, SQLProvider

def test_instantiation():
    print("Testing initialization...")
    
    db_manager = DatabaseManager()
    
    milvus = MilvusProvider({"uri": "http://localhost:19530"})
    neo4j = Neo4jProvider({"uri": "bolt://localhost:7687"})
    sql = SQLProvider({"database": "test.db"})
    
    db_manager.register_database("milvus", milvus)
    db_manager.register_database("neo4j", neo4j)
    db_manager.register_database("sql", sql)
    
    assert db_manager.get_provider("milvus") == milvus
    assert db_manager.get_provider("neo4j") == neo4j
    assert db_manager.get_provider("sql") == sql
    
    print("✅ All providers instantiated and registered successfully.")

if __name__ == "__main__":
    test_instantiation()
