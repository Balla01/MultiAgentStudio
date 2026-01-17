from typing import Dict, List, Optional, Any
import logging
from studio.core.agent_factory import AgentFactory
from studio.core.database_manager import DatabaseManager

logger = logging.getLogger(__name__)

class AgentService:
    def __init__(self, agent_factory: AgentFactory, db_manager: DatabaseManager):
        self.factory = agent_factory
        self.db_manager = db_manager
        self.active_agents: Dict[str, Any] = {}
        self.session_metadata: Dict[str, Dict] = {}

    def list_available_crews(self) -> List[str]:
        return self.factory.list_available_crews()

    def create_agent_session(self, session_id: str, crew_name: str, selected_dbs: List[str]) -> bool:
        """
        Create and store an agent instance with DB connections.
        """
        # Get DB providers
        db_connections = {}
        for db_name in selected_dbs:
            provider = self.db_manager.get_provider(db_name)
            if provider:
                db_connections[db_name] = provider
            else:
                logger.warning(f"Database provider '{db_name}' not found.")

        # Load crew
        crew = self.factory.load_crew(crew_name, db_connections)
        
        if crew:
            self.active_agents[session_id] = crew
            self.session_metadata[session_id] = {
                "crew_name": crew_name,
                "dbs": selected_dbs
            }
            logger.info(f"Created session {session_id} with crew {crew_name}")
            return True
        else:
            logger.error(f"Failed to create crew {crew_name} for session {session_id}")
            return False
    
    def get_agent(self, session_id: str) -> Any:
        return self.active_agents.get(session_id)

    def kickoff(self, session_id: str, inputs: Dict[str, Any]) -> Any:
        agent = self.get_agent(session_id)
        if not agent:
             raise ValueError(f"Session {session_id} not found")
        
        return agent.kickoff(inputs=inputs)
