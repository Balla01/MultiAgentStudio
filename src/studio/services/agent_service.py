from typing import Dict, List, Optional, Any
import logging
from studio.services.crew_factory import CrewFactory
from studio.core.database_manager import DatabaseManager

logger = logging.getLogger(__name__)

class AgentService:
    def __init__(self, crew_factory: CrewFactory, db_manager: DatabaseManager):
        self.factory = crew_factory
        self.db_manager = db_manager
        self.active_agents: Dict[str, Any] = {}
        self.session_metadata: Dict[str, Dict] = {}
        self.crew_cache: Dict[tuple, Any] = {}

    def list_available_crews(self) -> List[str]:
        return self.factory.list_available_crews()

    def create_agent_session(self, session_id: str, crew_name: str, selected_dbs: List[str], work_item_name: Optional[str] = None) -> bool:
        """
        Create and store an agent instance with DB connections.
        
        Args:
            session_id: Unique session identifier
            crew_name: Name of crew to load
            selected_dbs: List of databases to enable
            work_item_name: Optional work item for scoped memory
        """
        # Get DB providers
        db_connections = {}
        for db_name in selected_dbs:
            provider = self.db_manager.get_provider(db_name)
            if provider:
                db_connections[db_name] = provider
            else:
                logger.warning(f"Database provider '{db_name}' not found.")

        # Determine storage path for memory isolation
        storage_path = None
        if work_item_name:
            import os
            # Sanitize work item name for filesystem
            safe_name = "".join([c for c in work_item_name if c.isalnum() or c in ('-', '_')]).strip()
            storage_path = os.path.abspath(f"./storage/memory/{safe_name}")
            logger.info(f"Using isolated memory path for work item '{work_item_name}': {storage_path}")

        cache_key = (crew_name, tuple(sorted(selected_dbs)), work_item_name or "")

        # Reuse existing crew if configuration matches
        try:
            crew = self.crew_cache.get(cache_key)
            if crew is None:
                crew = self.factory.create_crew(
                    crew_name, 
                    db_connections=db_connections,
                    storage_path=storage_path
                )
                self.crew_cache[cache_key] = crew
                logger.info(f"Created and cached crew for key: {cache_key}")
            else:
                logger.info(f"Reusing cached crew for key: {cache_key}")
            
            if crew:
                self.active_agents[session_id] = crew
                self.session_metadata[session_id] = {
                    "crew_name": crew_name,
                    "dbs": selected_dbs,
                    "work_item": work_item_name # Store work item for reference
                }
                logger.info(f"Created session {session_id} with crew {crew_name}")
                return True
            else:
                logger.error(f"Failed to create crew {crew_name} for session {session_id}")
                return False
        except Exception as e:
            logger.error(f"Error creating crew session: {e}")
            return False
    
    def get_agent(self, session_id: str) -> Any:
        return self.active_agents.get(session_id)

    def kickoff(self, session_id: str, inputs: Dict[str, Any]) -> Any:
        agent = self.get_agent(session_id)
        if not agent:
             raise ValueError(f"Session {session_id} not found")
        
        return agent.kickoff(inputs=inputs)
