# rules_graph_ingest.py
import re
import json
import math
from neo4j import GraphDatabase
from typing import List, Dict, Any, Optional

# CONFIG
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "Rakesh@2001"
INPUT_FILE = "/home/ntlpt19/Documents/itf_rules.txt"   # path to your big text file

# ---------------------------
# Parsing utilities
# ---------------------------
def split_rule_blocks(text: str) -> List[str]:
    # Keep the "Rule X: id" as part of each block
    parts = re.split(r'(?m)^(Rule\s+\d+:\s+)', text)
    # re.split returns separators; recombine pairs
    blocks = []
    if len(parts) <= 1:
        return [text]
    it = iter(parts)
    prefix = next(it)  # may be preamble
    for sep in it:
        body = next(it, "")
        blocks.append(sep + body)
    return [b.strip() for b in blocks if b.strip()]

def simple_extract(pattern: str, text: str) -> Optional[str]:
    m = re.search(pattern, text, flags=re.IGNORECASE)
    return m.group(1).strip() if m else None

def extract_list(pattern: str, text: str) -> List[str]:
    m = re.search(pattern, text, flags=re.IGNORECASE)
    if not m:
        return []
    raw = m.group(1)
    # try to parse as JSON array or comma-separated
    try:
        return json.loads(raw.replace("'", "\""))
    except Exception:
        # fallback: split by comma and clean
        items = [x.strip().strip('[]"\'') for x in re.split(r',|\n', raw) if x.strip()]
        return [i for i in items if i]

def parse_rule_block(block: str) -> Optional[Dict[str, Any]]:
    # get rule id from "Rule <n>: <id>"
    m = re.search(r'Rule\s+\d+:\s*([^\n\r]+)', block, flags=re.IGNORECASE)
    if not m:
        return None
    rule_id = m.group(1).strip()

    rd = {
        "id": rule_id,
        "name": simple_extract(r'Rule Name:\s*(.*)', block),
        "short_name": simple_extract(r'Short Name:\s*(.*)', block),
        "group": simple_extract(r'Rule Group:\s*(.*)', block),
        "doc_type": simple_extract(r'Document Type:\s*(.*)', block),
        "product": simple_extract(r'Product:\s*(.*)', block),
        "type": simple_extract(r'\bType:\s*(.*)', block),
        "implementation": simple_extract(r'Implementation:\s*(.*)', block),
        "article": simple_extract(r'Rule Article:\s*(.*)', block),
        "source_documents": extract_list(r'Source Document\(s\):\s*(\[.*?\])', block),
        "dest_documents": extract_list(r'Destination Document\(s\):\s*(\[.*?\])', block),
        "source_keys": extract_list(r'Source Keys:\s*(\[.*?\])', block),
        "dest_keys": extract_list(r'Destination Keys:\s*(\[.*?\])', block),
        "condition": None,
        "operations": extract_list(r'Operation[s]?:\s*(\[.*?\])', block) or extract_list(r'Operation:\s*(\[.*?\])', block),
        "fail_reason": simple_extract(r'Fail Reasoning:\s*(.*)', block),
        "description": simple_extract(r'Rule Information:\s*(.*)', block) or simple_extract(r'How it works:\s*(.*)', block),
        "examples": []
    }

    # Try to extract JSON-like condition
    cond = simple_extract(r'Condition:\s*(\{.*\})', block)
    if cond:
        try:
            rd['condition'] = json.loads(cond.replace("'", '"'))
        except Exception:
            rd['condition'] = {"raw": cond}

    # Pull examples (PASS / FAIL lines) if present
    ex_lines = re.findall(r'([✓✗]\s*(PASS|FAIL|PASS:|FAIL:).*)', block)
    rd['examples'] = [line[0] for line in ex_lines] if ex_lines else []
    return rd

def parse_rules_file(path: str) -> List[Dict[str, Any]]:
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()
    blocks = split_rule_blocks(text)
    rules = []
    for b in blocks:
        parsed = parse_rule_block(b)
        if parsed:
            rules.append(parsed)
    return rules

# ---------------------------
# Neo4j loader (dump)
# ---------------------------
class RulesGraph:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def create_constraints(self):
        with self.driver.session() as s:
            s.run("CREATE CONSTRAINT IF NOT EXISTS FOR (r:Rule) REQUIRE r.id IS UNIQUE")
            s.run("CREATE CONSTRAINT IF NOT EXISTS FOR (d:DocumentType) REQUIRE d.name IS UNIQUE")
            s.run("CREATE CONSTRAINT IF NOT EXISTS FOR (f:Field) REQUIRE f.name IS UNIQUE")
            s.run("CREATE CONSTRAINT IF NOT EXISTS FOR (g:RuleGroup) REQUIRE g.name IS UNIQUE")
            s.run("CREATE CONSTRAINT IF NOT EXISTS FOR (p:Product) REQUIRE p.name IS UNIQUE")
            s.run("CREATE CONSTRAINT IF NOT EXISTS FOR (op:Operation) REQUIRE op.name IS UNIQUE")

    def ingest_rules(self, rules: List[Dict[str,Any]]):
        with self.driver.session() as s:
            # Delete existing nodes with same IDs to avoid duplicates/stale data
            rule_ids = [r['id'] for r in rules if 'id' in r]
            if rule_ids:
                print(f"Deleting {len(rule_ids)} existing rules before import...")
                s.run("MATCH (r:Rule) WHERE r.id IN $ids DETACH DELETE r", ids=rule_ids)

            for r in rules:
                s.execute_write(self._create_rule_tx, r)

    @staticmethod
    def _create_rule_tx(tx, rule):
        # Create Rule node with properties
        tx.run("""
        MERGE (r:Rule {id: $id})
        SET r.name = $name,
            r.short_name = $short_name,
            r.group = $group,
            r.product = $product,
            r.type = $type,
            r.implementation = $implementation,
            r.article = $article,
            r.description = $description,
            r.fail_reason = $fail_reason,
            r.condition_json = $condition_json,
            r.ops = $ops
        """, 
        id=rule['id'],
        name=rule.get('name'),
        short_name=rule.get('short_name'),
        group=rule.get('group'),
        product=rule.get('product'),
        type=rule.get('type'),
        implementation=rule.get('implementation'),
        article=rule.get('article'),
        description=rule.get('description'),
        fail_reason=rule.get('fail_reason'),
        condition_json=json.dumps(rule.get('condition')) if rule.get('condition') else None,
        ops=rule.get('operations') or []
        )

        # Group
        if rule.get('group'):
            tx.run("""
            MATCH (r:Rule {id:$id})
            MERGE (g:RuleGroup {name:$group})
            MERGE (r)-[:BELONGS_TO]->(g)
            """, id=rule['id'], group=rule['group'])

        # Product
        if rule.get('product'):
            tx.run("""
            MATCH (r:Rule {id:$id})
            MERGE (p:Product {name:$product})
            MERGE (r)-[:APPLIES_TO_PRODUCT]->(p)
            """, id=rule['id'], product=rule['product'])

        # Documents
        for doc in (rule.get('source_documents') or []):
            tx.run("""
            MATCH (r:Rule {id:$id})
            MERGE (d:DocumentType {name:$doc})
            MERGE (r)-[:APPLIES_TO]->(d)
            """, id=rule['id'], doc=doc)
        for doc in (rule.get('dest_documents') or []):
            tx.run("""
            MATCH (r:Rule {id:$id})
            MERGE (d:DocumentType {name:$doc})
            MERGE (r)-[:APPLIES_TO]->(d)
            """, id=rule['id'], doc=doc)

        # Fields
        for f in (rule.get('source_keys') or []):
            tx.run("""
            MATCH (r:Rule {id:$id})
            MERGE (fld:Field {name:$field})
            MERGE (r)-[:SOURCE_FIELD]->(fld)
            """, id=rule['id'], field=f)
        for f in (rule.get('dest_keys') or []):
            tx.run("""
            MATCH (r:Rule {id:$id})
            MERGE (fld:Field {name:$field})
            MERGE (r)-[:DEST_FIELD]->(fld)
            """, id=rule['id'], field=f)

        # Operations
        for op in (rule.get('operations') or []):
            tx.run("""
            MATCH (r:Rule {id:$id})
            MERGE (op:Operation {name:$op})
            MERGE (r)-[:USES_OPERATION]->(op)
            """, id=rule['id'], op=op)

        # Done
        print("Ingested:", rule['id'])

# ---------------------------
# Inference / Evaluation logic
# ---------------------------
def safe_parse_number(v):
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip().replace(',', '')
    # strip currency symbols
    s = re.sub(r'[^\d\.\-]', '', s)
    try:
        return float(s)
    except:
        return None

def evaluate_rule_against_documents(rule: Dict[str,Any], documents: Dict[str, Dict[str,Any]], compare_mode='max'):
    """
    rule: parsed rule dict (same shape as ingestion)
    documents: { "CI": {"invoice_amount": 10000, ...}, "BOE": {...}, ... }
    compare_mode: 'max' (default), 'all', or 'any'
    Returns: dict { 'rule_id', 'applicable': bool, 'passed': bool or None, 'details': {...} }
    """
    src_keys = rule.get('source_keys') or []
    dest_keys = rule.get('dest_keys') or []
    src_docs = rule.get('source_documents') or []
    dest_docs = rule.get('dest_documents') or []
    cond = (rule.get('condition') or {})
    op = cond.get('operation') or (rule.get('operations') or [None])[0]

    # currently support number comparison and string_match
    if not src_keys or not dest_keys:
        return {'rule_id':rule['id'], 'applicable': False, 'passed': None, 'details': 'no keys'}

    # For simplicity handle single source key (most rules are like that)
    src_key = src_keys[0]
    # gather source values from specified source documents
    src_values = []
    for sd in src_docs:
        dv = documents.get(sd, {}).get(src_key)
        if dv is not None:
            src_values.append(dv)
    if not src_values:
        return {'rule_id':rule['id'], 'applicable': False, 'passed': None, 'details':'source value missing'}

    # pick one source value (if multiple sources exist, choose first or make this logic configurable)
    src_val = src_values[0]
    src_num = safe_parse_number(src_val)

    # gather destination values across all destination docs and dest_keys
    dest_values_all = []
    for dd in dest_docs:
        doc = documents.get(dd, {})
        for dk in dest_keys:
            v = doc.get(dk)
            if v is not None:
                dest_values_all.append(v)
    if not dest_values_all:
        return {'rule_id':rule['id'], 'applicable': False, 'passed': None, 'details':'destination values missing'}

    # numeric comparison
    if op and op.lower() in ('lte', 'number_comparison'):
        dest_nums = [safe_parse_number(x) for x in dest_values_all]
        dest_nums = [x for x in dest_nums if x is not None]
        if src_num is None or not dest_nums:
            return {'rule_id':rule['id'], 'applicable': False, 'passed': None, 'details':'numeric parse failed'}
        if compare_mode == 'all':
            passed = all(src_num <= x for x in dest_nums)
        elif compare_mode == 'any':
            passed = any(src_num <= x for x in dest_nums)
        else:  # 'max' (default)
            passed = src_num <= max(dest_nums)
        return {'rule_id':rule['id'], 'applicable': True, 'passed': passed,
                'details': {'src': src_num, 'dest_nums': dest_nums, 'op': op, 'mode': compare_mode}}

    # string match
    if op and op.lower() in ('string_match', 'str_match'):
        # require that all document values equal (or use 'any'/'all' config)
        src_str = str(src_val).strip().lower()
        dest_strs = [str(x).strip().lower() for x in dest_values_all]
        # pass if src matches all destinations (change as needed)
        passed = all(src_str == ds for ds in dest_strs)
        return {'rule_id':rule['id'], 'applicable': True, 'passed': passed,
                'details': {'src': src_str, 'dests': dest_strs, 'op': op}}

    # fallback: not supported operation
    return {'rule_id':rule['id'], 'applicable': False, 'passed': None, 'details': f'unknown op {op}'}


# ---------------------------
# Helper: load rules from Neo4j for evaluation
# ---------------------------
def load_rules_from_neo4j(uri, user, password) -> List[Dict[str,Any]]:
    driver = GraphDatabase.driver(uri, auth=(user, password))
    rules = []
    with driver.session() as s:
        # Retrieve rule and connected docs/fields/ops
        res = s.run("""
        MATCH (r:Rule)
        OPTIONAL MATCH (r)-[:APPLIES_TO]->(d:DocumentType)
        OPTIONAL MATCH (r)-[:SOURCE_FIELD]->(sf:Field)
        OPTIONAL MATCH (r)-[:DEST_FIELD]->(df:Field)
        OPTIONAL MATCH (r)-[:USES_OPERATION]->(op:Operation)
        RETURN r.id as id, r.name as name, r.description as description, 
               collect(DISTINCT d.name) as docs,
               collect(DISTINCT sf.name) as src_keys,
               collect(DISTINCT df.name) as dest_keys,
               collect(DISTINCT op.name) as ops,
               r.condition_json as condition_json,
               r.fail_reason as fail_reason,
               r.group as group
        """)
        for rec in res:
            cond = None
            if rec["condition_json"]:
                try:
                    cond = json.loads(rec["condition_json"])
                except:
                    cond = {"raw": rec["condition_json"]}
            rules.append({
                "id": rec["id"],
                "name": rec["name"],
                "description": rec["description"],
                "source_documents": rec["docs"],  # note: both source/dest docs stored together in example ingestion
                "dest_documents": [],             # we'll map docs->keys in a richer model if required
                "source_keys": rec["src_keys"],
                "dest_keys": rec["dest_keys"],
                "operations": rec["ops"],
                "condition": cond,
                "fail_reason": rec["fail_reason"],
                "group": rec["group"]
            })
    driver.close()
    return rules

# ---------------------------
# CLI / main
# ---------------------------
def main_ingest():
    rules = parse_rules_file(INPUT_FILE)
    print("Parsed rules:", len(rules))
    g = RulesGraph(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    g.create_constraints()
    g.ingest_rules(rules)
    g.close()
    print("Ingest complete.")

def demo_evaluate(documents:Dict[str, Dict[str, Any]], compare_mode='max'):
    # load rules from neo4j or from parsed rules list (for demonstration we call neo4j)
    rules = load_rules_from_neo4j(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    results = []
    for r in rules:
        res = evaluate_rule_against_documents(r, documents, compare_mode=compare_mode)
        results.append(res)
    return results

if __name__ == "__main__":
    # example usage:
    # 1) ingest: python rules_graph_ingest.py  (ensure INPUT_FILE correct)
    # 2) evaluate: call demo_evaluate(...) from interactive or extend this script to load sample doc JSON
    main_ingest()
