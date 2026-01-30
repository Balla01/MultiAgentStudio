import logging
import sys
import os

# Ensure src is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from studio.core import DatabaseManager, MilvusProvider, Neo4jProvider, SQLProvider, ConfigManager
from studio.core.memory_config import MemoryConfigManager
from studio.services.crew_factory import CrewFactory
from studio.services import AgentService, RAGService, ChatService
from studio.data import DataIngestionPipeline
from studio.ui import create_app

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    logger.info("Starting Agentic AI Studio...")
    
    # 1. Load Config
    config = ConfigManager()

    # 1.5 Setup Memory Config
    memory_config = MemoryConfigManager(
        storage_dir="./storage/memory",
        embedding_model="Alibaba-NLP/gte-large-en-v1.5",
        enable_short_term=True,
        enable_long_term=True,
        enable_entity=True
    )
    
    # 2. Setup Database Manager
    db_manager = DatabaseManager()
    
    # Milvus
    milvus_conf = config.get_db_config('milvus')
    if milvus_conf:
        db_manager.register_database('milvus', MilvusProvider(milvus_conf))
    
    # Neo4j
    neo4j_conf = config.get_db_config('neo4j')
    if neo4j_conf:
        db_manager.register_database('neo4j', Neo4jProvider(neo4j_conf))
    
    # SQL
    sql_conf = config.get_db_config('mysql')
    if sql_conf:
        db_manager.register_database('mysql', SQLProvider(sql_conf))
        
    # 3. Setup Services
    agent_config = config.get_agent_config()
    crews_dir = agent_config.get('crews_directory', 'src/studio/crews')
    
    # Resolve relative path
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(base_dir, "../../"))
    abs_crews_dir = os.path.join(project_root, crews_dir)
    
    # Use CrewFactory with Memory
    crew_factory = CrewFactory(
        crews_directory=abs_crews_dir,
        memory_config=memory_config,
        db_manager=db_manager
    )
    
    # Update AgentService to use CrewFactory (Wait, AgentFactory is replaced by CrewFactory?)
    # The prompt implies CrewFactory replaces AgentFactory or AgentFactory logic is inside. 
    # But AgentService depends on AgentFactory currently.
    # Let's check AgentService signature. I will assume we should pass crew_factory instead of agent_factory if they have compatible interfaces
    # Or I should keep AgentFactory but maybe CrewFactory is better.
    # The User's prompt uses `from studio.services.crew_factory import CrewFactory` and creates `CrewFactory`.
    # And then passes `crew_factory` to `create_agent_selector`.
    # It does NOT instantiate AgentService in the example snippet! 
    # But current main.py does.
    # I should check if I need to update AgentService or if CrewFactory is enough for the UI components.
    # The UI components in the prompt:
    # create_agent_selector(crew_factory)
    # create_chat_interface(chat_service, ...)
    # create_memory_dashboard(crew_factory)
    # ChatService takes crew_factory?
    # "chat_service = ChatService(crew_factory, rag_service)" -> User prompt.
    
    # So I need to verify ChatService and update it if needed.
    # For now, I will use CrewFactory in main.py.

    # Update AgentService to use CrewFactory
    agent_service = AgentService(crew_factory, db_manager)
    
    rag_service = RAGService(db_manager)
    
    chat_service = ChatService(agent_service, rag_service) 
    
    # 4. Setup Data Pipeline
    pipeline = DataIngestionPipeline(db_manager)
    
    # 5. Build UI
    app = create_app(
        config,
        db_manager,
        pipeline,
        agent_service,
        chat_service,
        crew_factory
    )
    
    # 6. Launch
    logger.info("Global launch...")
    crew_factory.print_memory_summary()
    
    app.launch(
        server_name="0.0.0.0", 
        server_port=7860,
        share=False
    )

if __name__ == "__main__":
    main()
