import logging
import sys
import os

# Ensure src is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from studio.core import DatabaseManager, MilvusProvider, Neo4jProvider, SQLProvider, ConfigManager
from studio.core.agent_factory import AgentFactory
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
    # Assuming config path is relative to src/studio or project root.
    # We'll try to find it relative to project root usually
    project_root = os.path.abspath(os.path.join(base_dir, "../../"))
    abs_crews_dir = os.path.join(project_root, crews_dir)
    
    agent_factory = AgentFactory(crews_directory=abs_crews_dir)
    agent_service = AgentService(agent_factory, db_manager)
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
        agent_factory
    )
    
    # 6. Launch
    logger.info("Global launch...")
    app.launch(
        server_name="0.0.0.0", 
        server_port=7860,
        share=False
    )

if __name__ == "__main__":
    main()
