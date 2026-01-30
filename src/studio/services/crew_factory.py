"""
Factory for creating CrewAI crews with memory and custom embeddings.
Handles crew initialization with proper memory configuration.
"""

from crewai import Crew, Process
from typing import List, Dict, Any, Optional
import importlib
from pathlib import Path

from studio.core.memory_config import MemoryConfigManager
from studio.services.memory_monitor import MemoryEventMonitor
from studio.core.database_manager import DatabaseManager


class CrewFactory:
    """
    Factory for creating and managing CrewAI crews with memory support.
    """
    
    def __init__(
        self,
        crews_directory: str = "src/studio/crews",
        memory_config: Optional[MemoryConfigManager] = None,
        db_manager: Optional[DatabaseManager] = None
    ):
        """
        Initialize crew factory.
        
        Args:
            crews_directory: Path to crews directory
            memory_config: Memory configuration manager
            db_manager: Database manager for DB tools
        """
        self.crews_dir = Path(crews_directory)
        self.memory_config = memory_config or MemoryConfigManager()
        self.db_manager = db_manager
        
        # Initialize memory monitor
        self.memory_monitor = MemoryEventMonitor(
            log_to_file=True,
            log_dir="./logs"
        )
        
        print(f"✅ CrewFactory initialized")
        print(f"   Crews directory: {self.crews_dir}")
        print(f"   Memory enabled: Yes")
        print(f"   Embedding model: Alibaba-NLP/gte-large-en-v1.5")
    
    def list_available_crews(self) -> List[str]:
        """
        Scan crews directory and return available crew names.
        
        Returns:
            List of crew names
        """
        crews = []
        
        if not self.crews_dir.exists():
            print(f"⚠️  Crews directory not found: {self.crews_dir}")
            return crews
        
        for crew_path in self.crews_dir.iterdir():
            if crew_path.is_dir() and (crew_path / "crew.py").exists():
                crews.append(crew_path.name)
        
        return sorted(crews)
    
    def create_crew(
        self,
        crew_name: str,
        db_connections: Optional[Dict[str, Any]] = None,
        process: Process = Process.sequential,
        verbose: bool = True,
        enable_memory: bool = True,
        storage_path: Optional[str] = None
    ) -> Crew:
        """
        Create a crew instance with memory and custom embeddings.
        
        Args:
            crew_name: Name of the crew to load
            db_connections: Database connections for tools
            process: Crew process type
            verbose: Enable verbose output
            enable_memory: Enable memory system
            storage_path: Optional override for memory storage path (e.g., for work item isolation)
            
        Returns:
            Initialized Crew instance
        """
        # Load crew module
        try:
            module_path = f"studio.crews.{crew_name}.crew"
            crew_module = importlib.import_module(module_path)
        except ImportError as e:
            raise ValueError(f"Failed to load crew '{crew_name}': {e}")
        
        # Configure Mistral LLM
        # This replaces the default OpenAI LLM to avoid OPENAI_API_KEY requirement
        from crewai import LLM
        import os
        
        # Set dummy OPENAI_API_KEY to satisfy strict checks if any
        if not os.getenv("OPENAI_API_KEY"):
            os.environ["OPENAI_API_KEY"] = "NA"
        
        # Use env var or fallback (same as in utils.py)
        mistral_api_key = os.getenv("MISTRAL_API_KEY")
        if not mistral_api_key:
             print("⚠️ MISTRAL_API_KEY not found in environment. Agent execution may fail.")
             # We let it proceed as some users might rely on other mechanisms (e.g. Ollama/Local)
             # but for now we warn clearly.
        
        mistral_model = "mistral/mistral-large-latest"
        
        crew_llm = None
        try:
            if mistral_api_key:
                # Create shared LLM instance
                crew_llm = LLM(model=mistral_model, api_key=mistral_api_key)
            else:
                # If no key, maybe they want to use OpenAI or other? 
                # But our current requirement implies Mistral.
                pass
        except Exception as e:
            print(f"⚠️ Failed to configure Mistral LLM: {e}")

        # Get agents and tasks from crew module
        # Assume crew module has get_agents() and get_tasks() functions
        if hasattr(crew_module, 'get_agents'):
            agents = crew_module.get_agents(db_connections)
        else:
            raise ValueError(f"Crew '{crew_name}' missing get_agents() function")
        
        # Inject LLM into agents (FORCE OVERWRITE)
        if crew_llm:
            for agent in agents:
                agent.llm = crew_llm
            print(f"✅ Injected Mistral LLM ({mistral_model}) into {len(agents)} agents")
        
        if hasattr(crew_module, 'get_tasks'):
            tasks = crew_module.get_tasks(agents)
        else:
            raise ValueError(f"Crew '{crew_name}' missing get_tasks() function")
        
        # Create crew configuration
        crew_config = {
            "agents": agents,
            "tasks": tasks,
            "process": process,
            "verbose": verbose,
            "manager_llm": crew_llm, # Explicitly set manager LLM to Mistral
            "function_calling_llm": crew_llm # Explicitly set function calling LLM
        }
        
        # Add memory configuration if enabled
        if enable_memory:
            crew_config["memory"] = True
            
            # Determine memory config to use
            current_mem_config = self.memory_config
            
            # If storage_path provided, create a temporary scoped config
            if storage_path:
                print(f"🔒 Creating work-item scoped memory at: {storage_path}")
                # Import here to avoid circular dependencies if any, or just reuse class
                current_mem_config = MemoryConfigManager(storage_dir=storage_path)
            
            # Set storage dir in environment for CrewAI to pick up
            # Note: CrewAI reads CHROMA_DB_STORAGE_DIRECTORY environment variable or similar
            # But mostly it uses the embedder config location. 
            # Actually CrewAI 0.x uses local storage relative to execution or fixed paths unless configured.
            # We need to ensure we set the storage config correctly.
            
            # For CrewAI, setting the storage directory often happens via the embedding config 
            # OR by setting specific environment variables that CrewAI's memory system reads.
            # Let's check memory_config implementation to see what it sets.
            
            # Inject embedder config
            crew_config["embedder"] = current_mem_config.embedder_adapter.get_config()
            
            # We also need to set the environment variable for this specific run if possible,
            # but environment variables are process-wide. This might be a limitation if running parallel crews.
            # For now, sequential execution assumption allows this.
            os.environ["CREWAI_STORAGE_DIR"] = str(current_mem_config.storage_dir)
            
            
            # WORKAROUND: Set environment variable for ChromaDB
            os.environ["CHROMA_EMBEDDING_FUNCTION"] = "sentence-transformers"
            os.environ["CHROMA_EMBEDDING_MODEL"] = "Alibaba-NLP/gte-large-en-v1.5"
            
        crew_instance = Crew(**crew_config)
        
        # Register memory monitor with crew's event bus
        if enable_memory and hasattr(crew_instance, '_event_emitter'):
            self.memory_monitor.setup_listeners(crew_instance._event_emitter)
            print(f"✅ Memory monitoring enabled for crew '{crew_name}'")
            
        print(f"✅ Crew '{crew_name}' created successfully")
        print(f"   Agents: {len(agents)}")
        print(f"   Tasks: {len(tasks)}")
        print(f"   Memory: {'Enabled' if enable_memory else 'Disabled'}")
        
        return crew_instance

    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get memory statistics across all crews.
        
        Returns:
            Dictionary with memory statistics
        """
        return {
            'config': self.memory_config.get_memory_stats(),
            'operations': self.memory_monitor.get_statistics()
        }

    def print_memory_summary(self):
        """Print comprehensive memory summary"""
        print("\n" + "="*70)
        print("AGENTIC AI STUDIO - MEMORY SYSTEM STATUS")
        print("="*70)
        
        # Storage information
        print("\n📁 Storage Configuration:")
        config_stats = self.memory_config.get_memory_stats()
        print(f"   Location: {config_stats['storage_dir']}")
        print(f"   Exists: {config_stats['exists']}")
        
        if config_stats['memory_types']:
            print(f"\n   Memory Types:")
            for mem_type, info in config_stats['memory_types'].items():
                if 'size_mb' in info:
                    print(f"      {mem_type}: {info['size_mb']} MB")
        
        # Operation statistics
        print("\n📊 Operation Statistics:")
        self.memory_monitor.print_summary()

    def reset_memory(self, memory_type: str = 'all'):
        """
        Reset memory for all crews.
        
        Args:
            memory_type: Type of memory to reset ('short', 'long', 'entity', 'all')
        """
        self.memory_config.reset_memory(memory_type)
        print(f"✅ Memory reset: {memory_type}")


# Utility function for quick crew creation
def create_crew_with_memory(
    crew_name: str,
    crews_directory: str = "src/studio/crews",
    storage_dir: str = "./storage/memory",
    db_connections: Optional[Dict] = None
) -> Crew:
    """
    Quick utility to create a crew with memory enabled.
    
    Args:
        crew_name: Name of crew to create
        crews_directory: Path to crews directory
        storage_dir: Memory storage directory
        db_connections: Optional database connections
        
    Returns:
        Initialized Crew instance
    """
    memory_config = MemoryConfigManager(storage_dir=storage_dir)
    factory = CrewFactory(
        crews_directory=crews_directory,
        memory_config=memory_config
    )
    
    return factory.create_crew(
        crew_name=crew_name,
        db_connections=db_connections,
        enable_memory=True
    )
