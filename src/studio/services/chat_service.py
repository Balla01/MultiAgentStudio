from typing import List, Optional
import logging
from studio.services.agent_service import AgentService
from studio.services.rag_service import RAGService
from studio.core.utils import generate_answer_crew

logger = logging.getLogger(__name__)

class ChatService:
    def __init__(self, agent_service: AgentService, rag_service: RAGService):
        self.agent_service = agent_service
        self.rag_service = rag_service
        self.conversations = {}  # session_id -> history
    
    def chat(self, session_id: str, message: str, use_rag: bool = False, 
             enabled_dbs: List[str] = None, work_item_name: str = None) -> str:
        """
        Process chat message with optional RAG
        """
        if session_id not in self.conversations:
            self.conversations[session_id] = []
        
        context = ""
        # RAG retrieval if enabled
        if use_rag and enabled_dbs:
            logger.info(f"Retrieving context from {enabled_dbs} (Work Item: {work_item_name})...")
            try:
                retrieval_results = self.rag_service.retrieve(message, enabled_dbs, work_item_name=work_item_name)
                context = self.rag_service.assemble_context(retrieval_results)
            except Exception as e:
                logger.error(f"RAG failed: {e}")
                context = f"Error retrieving context: {e}"
        
        # Get agent
        try:
            agent = self.agent_service.get_agent(session_id)
            if not agent:
                return "Agent session not found. Please create a session first."
            response_str = generate_answer_crew(message, context)
            print("\n" + "="*20 + " FINAL Answer " + "="*20)
            print(response_str)
            # Prepare prompt
            full_message = message
            logger.info(f"Sending to agent: {full_message[:50]}...")

            # if context:
            #     full_message = f"Context:\n{context}\n\nUser Query: {message}"
            
            # # Kickoff Agent
            # response = agent.kickoff(inputs={'query': full_message})
            # response_str = str(response)
            
            # Store conversation
            self.conversations[session_id].append({
                'user': message,
                'assistant': response_str,
                'context': context if use_rag else None
            })
            
            return response_str
            
        except Exception as e:
            logger.error(f"Chat failed: {e}")
            return f"Error processing request: {str(e)}"
    
    def get_history(self, session_id: str):
        return self.conversations.get(session_id, [])
