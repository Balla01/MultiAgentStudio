import gradio as gr
import uuid
from studio.services.crew_factory import CrewFactory
from studio.services.agent_service import AgentService

def create_agent_selector(agent_service: AgentService):
    # agent_factory arg removed as agent_service handles it
    
    with gr.Column() as selector:
        gr.Markdown("## Agent Configuration")
        
        # Agent type selection
        agent_type = gr.Radio(
            choices=["Single Agent", "Multi-Agent (Crew)"],
            label="Agent Type",
            value="Multi-Agent (Crew)"
        )
        
        # Crew selection
        available_crews = agent_service.list_available_crews()
        crew_dropdown = gr.Dropdown(
            choices=available_crews,
            label="Select Crew",
            value=available_crews[0] if available_crews else None
        )
        
        # Database selection for agent
        agent_dbs = gr.CheckboxGroup(
            choices=["milvus", "neo4j", "mysql"],
            label="Connect Databases to Agent",
            value=["milvus"]
        )
        
        # RAG settings
        with gr.Accordion("RAG Settings", open=False):
            rag_enabled = gr.Checkbox(label="Enable RAG", value=True)
            top_k = gr.Slider(1, 20, value=5, step=1, label="Top K Results")
        
        create_btn = gr.Button("Create Agent Session", variant="primary")
        session_output = gr.Textbox(label="Session Info", lines=3)
        session_state = gr.State(value=None)
        
        def create_session(crew_name, selected_dbs):
            if not crew_name:
                return "❌ No crew selected", None
            
            session_id = str(uuid.uuid4())[:8]
            success = agent_service.create_agent_session(session_id, crew_name, selected_dbs)
            
            if success:
                return (
                    f"✅ Session Created: {session_id}\n"
                    f"Crew: {crew_name}\n"
                    f"Databases: {', '.join(selected_dbs)}",
                    session_id
                )
            else:
                return "❌ Failed to create session", None
        
        create_btn.click(
            create_session,
            inputs=[crew_dropdown, agent_dbs],
            outputs=[session_output, session_state]
        )
    
    return selector, session_state, rag_enabled, agent_dbs
