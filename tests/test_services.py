import sys
import os
from unittest.mock import MagicMock

# Add src to pythonpath
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from studio.core import DatabaseManager
from studio.core.agent_factory import AgentFactory
from studio.services import AgentService, RAGService, ChatService

class MockProvider:
    def query(self, params):
        if "query_text" in params:
            return [{"text": "Milvus Result", "score": 0.9}]
        if "entities" in params:
            return [{"r.name": "Neo4j Rule", "description": "Graph Desc"}]
        if "natural_query" in params:
            return {"data": [["SQL Result", 100]], "columns": ["name", "val"]}
        return []

class MockCrew:
    def kickoff(self, inputs):
        return f"Agent Answer based on: {inputs['query'][:20]}..."

def test_services():
    print("Testing Services...")
    
    # Setup
    db_manager = DatabaseManager()
    db_manager.register_database("milvus", MockProvider())
    db_manager.register_database("neo4j", MockProvider())
    db_manager.register_database("mysql", MockProvider())
    
    # RAG Service Test
    rag_service = RAGService(db_manager)
    results = rag_service.retrieve("test query", ["milvus", "neo4j", "mysql"])
    assert "milvus" in results
    assert "neo4j" in results
    assert "mysql" in results
    
    context = rag_service.assemble_context(results)
    print(f"RAG Context:\n{context}")
    assert "Milvus Result" in context
    assert "Graph Desc" in context
    assert "SQL Result" in context
    print("✅ RAG Service verified.")
    
    # Chat Service Test
    agent_factory = MagicMock()
    agent_factory.load_crew.return_value = MockCrew()
    agent_service = AgentService(agent_factory, db_manager)
    # Manually inject agent for session
    agent_service.active_agents["sess_001"] = MockCrew()
    
    chat_service = ChatService(agent_service, rag_service)
    
    # Test RAG Chat
    response = chat_service.chat("sess_001", "test query", use_rag=True, enabled_dbs=["milvus"])
    print(f"Chat Response: {response}")
    assert "Agent Answer" in response
    
    # Check history
    history = chat_service.get_history("sess_001")
    assert len(history) == 1
    assert history[0]['user'] == "test query"
    
    print("✅ Chat Service verified.")

if __name__ == "__main__":
    test_services()
