import sys
import os

# Add src to pythonpath
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from studio.core import DatabaseManager
from studio.core.agent_factory import AgentFactory
from studio.services import AgentService

class MockProvider:
    pass

def test_agent_service():
    print("Testing Agent Service...")
    
    # Setup
    db_manager = DatabaseManager()
    db_manager.register_database("milvus", MockProvider())
    
    # Point factory to local src/studio/crews
    crews_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../src/studio/crews"))
    factory = AgentFactory(crews_dir)
    service = AgentService(factory, db_manager)
    
    # List Crews
    crews = service.list_available_crews()
    print(f"Available crews: {crews}")
    assert "test_crew" in crews
    
    # Create Session
    session_id = "sess_001"
    success = service.create_agent_session(session_id, "test_crew", ["milvus"])
    assert success
    print("✅ Session created.")
    
    # Kickoff
    result = service.kickoff(session_id, inputs={"topic": "AI"})
    print(f"Result: {result}")
    assert "Test Crew executed" in str(result)
    
    print("✅ Agent Service verified.")

if __name__ == "__main__":
    test_agent_service()
