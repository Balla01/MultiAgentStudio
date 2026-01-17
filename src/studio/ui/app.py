import gradio as gr
from studio.core.config_manager import ConfigManager
from studio.core.database_manager import DatabaseManager
from studio.data import DataIngestionPipeline
from studio.core.agent_factory import AgentFactory
from studio.services import AgentService, RAGService, ChatService
from studio.ui.components.data_uploader import create_data_uploader
from studio.ui.components.agent_selector import create_agent_selector
from studio.ui.components.chat_interface import create_chat_interface

def create_app(config: ConfigManager, 
               db_manager: DatabaseManager,
               pipeline: DataIngestionPipeline,
               agent_service: AgentService,
               chat_service: ChatService,
               agent_factory: AgentFactory):

    ui_config = config.get_ui_config()
    
    with gr.Blocks(
        title=ui_config.get('title', "Agentic AI Studio"),
        theme=gr.themes.Soft()
    ) as demo:
        
        gr.Markdown(f"# 🤖 {ui_config.get('title', 'Agentic AI Studio')}")
        gr.Markdown("Build and deploy multi-agent AI systems with RAG")
        
        with gr.Tab("📁 Data Setup"):
            create_data_uploader(pipeline)
        
        with gr.Tab("⚙️ Agent Configuration"):
            agent_selector_col, session_state, rag_enabled, agent_dbs = create_agent_selector(
                agent_service, agent_factory
            )
        
        with gr.Tab("💬 Chat"):
            create_chat_interface(chat_service, session_state, rag_enabled, agent_dbs)
            
    return demo
