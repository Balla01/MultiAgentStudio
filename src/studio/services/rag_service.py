from typing import Dict, List, Any
import logging
from studio.core.database_manager import DatabaseManager
from studio.core.utils import milvus_inference
logger = logging.getLogger(__name__)

class RAGService:
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
    
    def retrieve(self, query: str, enabled_dbs: List[str], work_item_name: str = None, top_k: int = 5) -> Dict[str, Any]:
        """
        Query multiple databases and merge results
        """
        results = {}
        
        # Milvus: Semantic similarity search
        if 'milvus' in enabled_dbs:
            # work_item_name = 'itf_testing' # Now passed as arg
            if work_item_name:
                res = milvus_inference(query, work_item_name)
                results['milvus'] = res
            else:
                logger.warning("Milvus enabled but no work_item_name provided.")
                results['milvus'] = [] # Or handle error
        
        # Neo4j: Graph traversal
        if 'neo4j' in enabled_dbs:
             provider = self.db_manager.get_provider('neo4j')
             if provider:
                 # Extract potential entities (simple heuristic: generic nouns or split by space)
                 # For better results, use an LLM or NER model
                 entities = [w for w in query.split() if len(w) > 3]
                 res = provider.query({
                     'entities': entities,
                     'limit': top_k
                 })
                 results['neo4j'] = res
        
        # MySQL/SQLite: Keyword/SQL search
        if 'mysql' in enabled_dbs:
             provider = self.db_manager.get_provider('mysql')
             if provider:
                 # Text-to-SQL
                 res = provider.query({
                     'natural_query': query
                 })
                 results['mysql'] = res
        
        return results
    
    def assemble_context(self, retrieval_results: Dict[str, Any]) -> str:
        """Merge and rerank results from multiple DBs"""
        context_parts = []
        
        if 'milvus' in retrieval_results and retrieval_results['milvus']:
            context_parts.append("=== Milvus Semantic Search Results ===")
            # retrieval_results['milvus'] is now a list of chunks (dicts)
            for item in retrieval_results['milvus']:
                text = item.get('text', '')
                score = item.get('rerank_score', item.get('retrieval_score', 0))
                context_parts.append(f"[Score: {score:.4f}] {text}")
        
        if 'neo4j' in retrieval_results:
             context_parts.append("\n=== Graph Knowledge ===")
             for item in retrieval_results['neo4j']:
                 # Format graph node/rule
                 name = item.get('r.name', 'Unknown')
                 desc = item.get('r.description', item.get('description', ''))
                 context_parts.append(f"Rule: {name}\nDetails: {desc}")

        if 'mysql' in retrieval_results:
            context_parts.append("\n=== Structured Data ===")
            res = retrieval_results['mysql']
            if 'error' not in res and 'data' in res:
                cols = res.get('columns', [])
                for row in res.get('data', [])[:5]: # Limit to 5 rows
                    row_str = ", ".join([f"{c}: {v}" for c, v in zip(cols, row)])
                    context_parts.append(row_str)
            elif 'error' in res:
                context_parts.append(f"Query Error: {res['error']}")
        
        return "\n\n".join(context_parts)
