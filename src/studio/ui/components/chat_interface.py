import gradio as gr
from studio.services.chat_service import ChatService

def create_chat_interface(chat_service: ChatService, session_state, rag_enabled, agent_dbs):
    
    with gr.Column() as chat:
        gr.Markdown("## Chat Interface")
        
        chatbot = gr.Chatbot(
            label="Conversation",
            height=500,
            type="messages"
        )
        
        with gr.Row():
            msg_input = gr.Textbox(
                label="Message",
                placeholder="Ask anything...",
                scale=4
            )
            send_btn = gr.Button("Send", variant="primary", scale=1)
        
        with gr.Row():
            clear_btn = gr.Button("Clear History")
            
            with gr.Accordion("Debug Info", open=False):
                query_info = gr.JSON(label="Last Query Info")
        
        with gr.Row():
            work_item_id = gr.Textbox(
                label="Work Item ID (Milvus Collection)",
                placeholder="e.g., itf_testing can be left empty if not using Milvus",
                scale=2
            )
        
        def respond(message, history, session_id, use_rag, enabled_dbs, wid):
            if not session_id:
                history.append({"role": "user", "content": message})
                history.append({"role": "assistant", "content": "⚠️ Please create an agent session first in the 'Agent Configuration' tab."})
                return history, ""
            
            # Update history with user message immediately
            history.append({"role": "user", "content": message})
            
            # Get response from chat service
            response = chat_service.chat(
                session_id=session_id,
                message=message,
                use_rag=use_rag,
                enabled_dbs=enabled_dbs if use_rag else None,
                work_item_name=wid
            )
            
            history.append({"role": "assistant", "content": response})
            return history, ""
        
        def clear_history():
            return []
        
        send_btn.click(
            respond,
            inputs=[msg_input, chatbot, session_state, rag_enabled, agent_dbs, work_item_id],
            outputs=[chatbot, msg_input]
        )
        
        msg_input.submit(
            respond,
            inputs=[msg_input, chatbot, session_state, rag_enabled, agent_dbs, work_item_id],
            outputs=[chatbot, msg_input]
        )
        
        clear_btn.click(clear_history, outputs=chatbot)
    
    return chat
