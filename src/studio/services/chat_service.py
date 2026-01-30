from typing import List, Optional
import logging
from studio.services.agent_service import AgentService
from studio.services.rag_service import RAGService
from studio.core.utils import generate_answer_crew
from studio.services.memory_manager import retrieve_history
logger = logging.getLogger(__name__)

class ChatService:
    def __init__(self, agent_service: AgentService, rag_service: RAGService):
        self.agent_service = agent_service
        self.rag_service = rag_service
        self.conversations = {}  # session_id -> history
    
    def chat(self, session_id: str, message: str, use_rag: bool = False, 
             enabled_dbs: List[str] = None, work_item_name: str = None,
             use_history: bool = True) -> str:
        """
        Process chat message with optional RAG and persistent memory.
        """
        if session_id not in self.conversations:
            self.conversations[session_id] = []
        
        retrieval_context = ""
        # RAG retrieval if enabled
        if use_rag and enabled_dbs:
            logger.info(f"Retrieving context from {enabled_dbs} (Work Item: {work_item_name})...")
            try:
                # Assuming retrieval returns a list of chunks/documents
                retrieval_results = self.rag_service.retrieve(message, enabled_dbs, work_item_name=work_item_name)
                # Formatted context handling scores
                retrieval_context = self.rag_service.assemble_context(retrieval_results)
            except Exception as e:
                logger.error(f"RAG failed: {e}")
                retrieval_context = f"Error retrieving context: {e}"
        
        # Get memory-enabled agent
        try:
            agent = self.agent_service.get_agent(session_id)
            if not agent:
                return "Agent session not found. Please create a session first."
            
            response_str = generate_answer_crew(message, retrieval_context)
            if not use_history:
                return response_str
            # --- MEMORY RETRIEVAL FOR PROMPT ---
            # Native CrewAI memory is accessed via agent.memory if using single agent or crew structure
            # For a Crew instance, we access the underlying agents or crew memory
            
            # Note: CrewAI separates memory by ShortTerm, LongTerm, Entity
            # We will rely on CrewAI's internal memory mechanism during kickoff, 
            # BUT the user wants precise control over the prompt structure.
            
            # To get "Relevant past conversation", we can fetch from ShortTermMemory
            # Since accessing internal memory object might be complex or vary by version,
            # we will rely on the session history we store, OR try to fetch from agent if possible.
            # Using self.conversations currently for "Relevant past conversation" as a reliable fallback.
            
            history_text = "History usage disabled by user."
            if use_history:
                response_str = retrieve_history(message)#response_str ==> retrival answer?
                return response_str
                # Constructing history string from last 5 interactions
                recent_history = self.conversations.get(session_id, [])[-5:]
                history_text = ""
                for turn in recent_history:
                    history_text += f"User: {turn['user']}\nAssistant: {turn['assistant']}\n\n"
                
                if not history_text:
                    history_text = "No previous conversation."

            # Constructing the Custom Prompt
            custom_prompt_task = f"""
                You are given:
                - 'short': 'short_term_memory'
                - 'long': 'long_term_memory'
                - 'entity': 'entities'
                - 'knowledge': 'knowledge'

                3. The current user question.

                INSTRUCTIONS:
                - Carefully check whether the retrieved past conversation is relevant to the current question.
                - Use the retrieved past conversation ONLY if it directly helps answer the current question.
                - If the retrieved history is partially relevant, use only the relevant parts.
                - Do NOT invent details that are not present in the retrieved history.
                - If the answer cannot be fully derived from the retrieved history, answer using general domain knowledge and clearly explain your reasoning.
                - Give priority to factual accuracy and regulatory correctness.

                Relevant past conversation:
                {history_text}

                Recent interaction:
                (See above history)

                Retrieval Results:
                {retrieval_context}

                Current question:
                {message}

                Answer the question clearly and concisely.
                Note: If retrieval results are present, cite them where appropriate.
                """
            # Logging for debug
            logger.info("Sending prompt to Memory-Enabled Agent...")
            
            # kickoff execution
            # CrewAI kickoff takes a 'inputs' dict which is usually interpolated into the Task description.
            # However, since we defined the prompt explicitly above, we want to run this specific instruction.
            # The 'test_crew' assumes tasks are pre-defined. 
            # To force this specific prompt, we can pass it as a special input OR 
            # if we standardized the crew to accept 'query' and 'context', we might need to adjust.
            
            # For this "Advanced" mode where the PROMPT is dynamic, we really want to replace the current task description
            # with our custom prompt.
            # Hack: We can update the first task's description dynamically.
            
            if hasattr(agent, 'tasks') and agent.tasks:
                agent.tasks[0].description = custom_prompt_task
                agent.tasks[0].expected_output = "A clear and concise answer based on context and history."
            
            # Execute
            response_obj = agent.kickoff()
            response_str = str(response_obj)
            
            print("\n" + "="*20 + " FINAL Answer " + "="*20)
            print(response_str)
            
            # Store conversation
            self.conversations[session_id].append({
                'user': message,
                'assistant': response_str,
                'context': retrieval_context if use_rag else None
            })
            
            return response_str
            
        except Exception as e:
            logger.error(f"Chat failed: {e}")
            import traceback
            traceback.print_exc()
            return f"Error processing request: {str(e)}"
    
    def get_history(self, session_id: str):
        return self.conversations.get(session_id, [])
