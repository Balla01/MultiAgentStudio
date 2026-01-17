# File: /home/ntlpt19/personal_projects/CompanyResearch/neo4j_with_vector/graph_inference.py

from neo4j import GraphDatabase
from crewai import Agent, Task, Crew, Process, LLM
import re
import json
from typing import List, Dict, Any
from difflib import SequenceMatcher

# --- Setup Neo4j driver ---
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "Rakesh@2001"

STOPWORDS = {
    "what", "is", "the", "a", "an", "please", "show", "tell", "me",
    "rule", "info", "about", "of", "in", "for", "how", "to", "get"
}

def normalize_text(s: str) -> str:
    return (s or "").strip()

def extract_tokens(query: str):
    tokens = re.findall(r'\b[a-zA-Z0-9_]+\b', query.lower())
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]

def find_id_candidate(tokens: list):
    for t in tokens:
        if '_' in t and any(ch.isdigit() for ch in t):
            return t
    for t in tokens:
        if re.match(r'^[a-z]{2,}_[a-z0-9_]+_\d+$', t):
            return t
    return None

def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, (a or "").lower(), (b or "").lower()).ratio()

def format_record(rec, score=None):
    return {
        "rule_id": rec.get("rule_id") or rec.get("id"),
        "name": rec.get("name"),
        "description": rec.get("desc") or rec.get("description"),
        "documents": rec.get("documents") or rec.get("docs") or [],
        "fields": rec.get("fields") or [],
        "fail_reason": rec.get("fail_reason"),
        "condition": rec.get("condition"),
        "score": score if score is not None else 0.0
    }

# --- Setup LLM ---
llm = LLM(model="mistral/mistral-large-latest", temperature=0.7, api_key="0TD9nsBifR6Lkr1kOag9aikbCBImYfGg")

# --- LLM for Extraction ---
def extract_search_terms(query: str) -> List[str]:
    prompt = f"""
    You are a search query analyzer. Extract the most important keywords, entity names, function names, rule IDs, document types, or field names from the user's query.
    Return ONLY a raw JSON list of strings. Do not include 'json' markdown fencing.

    User Query: "{query}"

    Keywords (JSON List):
    """
    response = llm.call(prompt)
    try:
        # cleanup markdown if present
        clean = response.strip().replace("```json", "").replace("```", "")
        return json.loads(clean)
    except:
        return extract_tokens(query)

class RuleGraphRetriever:
    def __init__(self):
        self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    def close(self):
        self.driver.close()

    def get_deep_context(self, user_query: str) -> list:
        # Step 1: LLM Keyword Extraction
        search_terms = extract_search_terms(user_query)
        print(f"LLM Extracted Terms: {search_terms}")
        
        results_map = {} # ID -> formatted_dict

        with self.driver.session() as session:
            for term in search_terms:
                if len(term) < 2: continue
                # Step 2: Find Anchor and Step 3: Expand Neighbors (2 Hops)
                # We want: (Node) --[1..2]-- (Rule)
                # Or just (Rule) checks itself.
                print(f"Expanding term: {term}")
                query = """
                    MATCH (n)
                    WHERE toLower(n.name) CONTAINS $term 
                       OR toLower(n.id) CONTAINS $term
                       OR toLower(n.description) CONTAINS $term
                    
                    // Traverse 1 or 2 hops to find relevant Rule nodes
                    // Pattern: (n)-[*0..2]-(r:Rule)
                    // We specifically want 'Rules' in the context
                    CALL apoc.path.subgraphNodes(n, {
                        maxLevel: 2,
                        labelFilter: '+Rule'
                    }) YIELD node AS r
                    
                    // For each found Rule, get its immediate context
                    OPTIONAL MATCH (r)-[:APPLIES_TO]->(d:DocumentType)
                    OPTIONAL MATCH (r)-[:SOURCE_FIELD|DEST_FIELD]->(f:Field)
                    OPTIONAL MATCH (r)-[:USES_OPERATION]->(op:Operation)
                    
                    RETURN DISTINCT r.id AS rule_id, 
                           r.name AS name, 
                           r.description AS description, 
                           collect(DISTINCT d.name) AS documents,
                           collect(DISTINCT f.name) AS fields,
                           collect(DISTINCT op.name) AS ops,
                           r.fail_reason AS fail_reason,
                           r.condition_json AS condition
                    LIMIT 20
                """
                # Fallback if APOC not available: Pure Cypher
                # We'll use a Union or simpler path match.
                # Let's assume APOC might key fail, so use pure Cypher Path
                
                cypher_pure = """
                    MATCH (n)
                    WHERE toLower(n.name) CONTAINS $term 
                       OR toLower(n.id) CONTAINS $term
                       OR (n:Rule AND toLower(n.description) CONTAINS $term)
                    
                    WITH n
                    MATCH (n)-[*0..2]-(r:Rule)
                    
                    // Deduplicate Rules per term
                    WITH DISTINCT r
                    
                    OPTIONAL MATCH (r)-[:APPLIES_TO]->(d:DocumentType)
                    OPTIONAL MATCH (r)-[:SOURCE_FIELD|DEST_FIELD]->(f:Field)
                    OPTIONAL MATCH (r)-[:USES_OPERATION]->(op:Operation)
                    
                    RETURN r.id AS rule_id, 
                           r.name AS name, 
                           r.description AS description, 
                           collect(DISTINCT d.name) AS documents,
                           collect(DISTINCT f.name) AS fields,
                           collect(DISTINCT op.name) AS ops,
                           r.fail_reason AS fail_reason,
                           r.condition_json AS condition
                    LIMIT 20
                """
                
                try:
                    res = session.run(cypher_pure, {"term": term.lower()})
                    count = 0
                    for record in res:
                        count += 1
                        rd = dict(record)
                        rid = rd['rule_id']
                        if rid and rid not in results_map:
                             # Format slightly differently including ops
                             results_map[rid] = {
                                "rule_id": rid,
                                "name": rd['name'],
                                "description": rd['description'],
                                "documents": rd['documents'],
                                "fields": rd['fields'],
                                "ops": rd['ops'],
                                "fail_reason": rd['fail_reason'],
                                "condition": rd['condition']
                             }
                    print(f"  -> Found {count} rules for term '{term}'")
                except Exception as e:
                    print(f"Error expanding term '{term}': {e}")
        
        return list(results_map.values())

# --- Inference function ---
def answer_user_query(query: str):
    graph = RuleGraphRetriever()
    context = graph.get_deep_context(query)
    graph.close()

    if not context:
        return f"No relevant rules found in the graph database for query: '{query}'"

    # Summarize larger context
    rule_summaries = []
    for r in context[:15]: # Limit to top 15 to fit context window
        s = f"Rule ID: {r['rule_id']}\nName: {r['name']}\nDescription: {r['description']}\n"
        if r['documents']: s += f"Docs: {r['documents']}\n"
        if r['fields']: s += f"Fields: {r['fields']}\n"
        if r.get('ops'): s += f"Ops: {r['ops']}\n"
        s += f"Fail: {r['fail_reason']}\nCond: {r['condition']}"
        rule_summaries.append(s)
    
    context_str = "\n---\n".join(rule_summaries)

    prompt = f"""
You are a regulatory trade-finance assistant. Answer the user's question based ONLY on the provided Graph Context.
Count carefully if asked.

Graph Context (Retrieved Rules):
{context_str}

User Question: "{query}"

Answer:
"""
    print("==========================")
    print(f"Generating Answer with {len(context)} rules in context...")
    print("==========================")
    response = llm.call(prompt)
    return response

# Example:
if __name__ == "__main__":
    user_question = "how many rules are using the function keyword_contains_in_ocr_data ?"
    final_answer = answer_user_query(user_question)
    print("\nFinal Answer:\n", final_answer)