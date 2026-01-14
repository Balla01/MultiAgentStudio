import ollama
import json
import re

# -----------------------------
# Input Text
# -----------------------------
org_text = """
ICC Uniform Rules for Demand Guarantees (URDG 758)

Article 1 - Application of URDG

The Uniform Rules for Demand Guarantees ("URDG") apply to any demand guarantee or
counter-guarantee that expressly indicates it is subject to them. They are binding on all parties
to the demand guarantee or counter-guarantee except so far as the demand guarantee or
counter-guarantee modifies or excludes them.

Where, at the request of the counter-guarantor, a demand guarantee is issued subject to the
URDG, the counter-guarantee shall also be subject to the URDG, unless the counter-guarantee
excludes the URDG. However, a demand guarantee does not become subject to the URDG
merely because the counter-guarantee is subject to the URDG.

Where, at the request or with the agreement of the instructing party, a demand guarantee or
counter-guarantee is issued subject to the URDG, the instructing party is deemed to have agreed.
"""

# -----------------------------
# Ollama Call Helper
# -----------------------------
def call_ollama(prompt, use_json=False):
    options = {
        "temperature": 0.1,   # low temp = better structure
        # "top_p": 0.9,
        # "num_ctx": 2048
    }
    
    kwargs = {
        "model": "llama3.2:1b",
        "prompt": prompt,
        "options": options,
        "stream": False
    }
    
    if use_json:
        kwargs["format"] = "json"

    try:
        response = ollama.generate(**kwargs)
        return response["response"].strip()
    except Exception as e:
        print(f"Error calling Ollama: {e}")
        return "{}"

# -----------------------------
# Safe JSON Loader
# -----------------------------
def safe_json_load(text, expected_type=list):
    try:
        # Regex to clean potential markdown wrappers
        clean_text = re.sub(r'```json\s*|\s*```', '', text).strip()
        data = json.loads(clean_text)
        if isinstance(data, dict):
            # If model returns {"keywords": [...]}, extract the list
            if len(data) == 1:
                return list(data.values())[0]
            # If generic dict, return it if expected dict, else keys? 
            # Ideally prompt ensures list, but model might wrap in object
        return data if isinstance(data, expected_type) else []
    except json.JSONDecodeError:
        # Fallback: try to find list brackets
        start = text.find('[')
        end = text.rfind(']')
        if start != -1 and end != -1:
            try:
                candidate = json.loads(text[start:end+1])
                if isinstance(candidate, expected_type):
                    return candidate
            except:
                pass
        return []

# -----------------------------
# 1. Extract Keywords
# -----------------------------
keywords_prompt = f"""
Analyze the text and extract a list of precise domain-specific keywords and technical terms.

Text:
{org_text}

Rules:
1. Output MUST be valid JSON.
2. The format MUST be a list of strings. Example: ["term 1", "term 2", "term 3"]
3. Normalize to lowercase unless it's a proper noun (e.g. "URDG").
4. No introductory text.
"""

print("Generating Keywords...")
keywords_raw = call_ollama(keywords_prompt, use_json=True)
keywords = keywords_raw #safe_json_load(keywords_raw, list)

# -----------------------------
# 2. Extract Entities
# -----------------------------
entities_prompt = f"""
Analyze the text and extract a list of precise legal entities, organizations, document titles, and roles.

Text:
{org_text}

Rules:
1. Output MUST be valid JSON.
2. The format MUST be a list of strings. Example: ["Entity Name", "Document Title", "Role"]
3. Extract exact names as they appear in the text.
4. No introductory text.
"""

print("Generating Entities...")
entities_raw = call_ollama(entities_prompt, use_json=True)
entities = entities_raw #safe_json_load(entities_raw, list)

# -----------------------------
# 3. Generate Summary
# -----------------------------
summary_prompt = f"""
- Summarize the main rule or policy described in the text below.
- capturing the core application and binding nature of the rules.
- Do not add information not present in the text.
- Return ONLY plain text (not JSON).

Text:
{org_text}
"""

print("Generating Summary...")
summary = call_ollama(summary_prompt, use_json=False)

# -----------------------------
# 4. Combine Final Metadata
# -----------------------------
final_metadata = {
    "keywords": keywords,
    "summary": summary,
    "entities": entities
}

print(json.dumps(final_metadata, indent=2))
