# =============================================
# crewai_milvus_search.py
# =============================================

from pymilvus import connections, Collection
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import tool
from sentence_transformers import SentenceTransformer
# =============================================
# 1. Connect to Milvus
# =============================================
connections.connect(alias="default", host="localhost", port="19530")
print("✅ Connected to Milvus!")

sentence_transformer_ef = SentenceTransformer("all-MiniLM-L6-v2")
collection_name = "my_text_collection"  # existing Milvus collection name


# =============================================
# 2. Define Search Tool
# =============================================
@tool
def milvus_search(collection_name: str, query_text: str, top_k: int = 3) -> str:
    """Embed a query and search similar entries in Milvus."""
    # query_vector = model_sen.encode(query_text).tolist()
    # Encode the query text
    query_vector = sentence_transformer_ef.encode(query_text).tolist()

    collection = Collection(collection_name)
    collection.load()

    results = collection.search(
        data=[query_vector],
        anns_field="vector",
        param={"metric_type": "L2", "params": {"nprobe": 10}},
        limit=top_k,
        output_fields=["id"],
    )

    output = []
    for hits in results:
        for hit in hits:
            output.append(f"id={hit.id}, distance={hit.distance:.4f}")
    return "\n".join(output)


# =============================================
# 3. Define LLM and Agent
# =============================================
# MISTRAL_API_KEY=
llm = LLM(model="mistral/mistral-large-latest", temperature=0.7, api_key="0TD9nsBifR6Lkr1kOag9aikbCBImYfGg")

milvus_agent = Agent(
    role="Milvus Semantic Search Agent",
    goal="Retrieve and summarize trade-related information from Milvus.",
    backstory=(
        "You specialize in searching semantic embeddings in Milvus and explaining "
        "the most relevant trade finance rules or regulations found."
    ),
    tools=[milvus_search],
    llm=llm,
    verbose=True,
)

# =============================================
# 4. Define Search Task
# =============================================
def build_search_task(query: str) -> Task:
    """Dynamically build a search task for any given query."""
    return Task(
        description=(
            f"Search the '{collection_name}' collection in Milvus for information related to:\n"
            f"'{query}'\n"
            "Use the milvus_search tool to retrieve top similar embeddings. "
            "Then summarize the most relevant information you find."
        ),
        expected_output="A concise summary of the information retrieved from Milvus relevant to the query.",
        agent=milvus_agent,
    )
# =============================================
# 5. Run Search for Any Query
# =============================================
def run_query(query: str):
    """Execute a CrewAI search for a given query string."""
    search_task = build_search_task(query)
    crew = Crew(
        agents=[milvus_agent],
        tasks=[search_task],
        process=Process.sequential,
        verbose=True,
    )

    result = crew.kickoff()
    print(f"\n=== ✅ RESULTS for query: '{query}' ===\n")
    print(result)
    return result


# =============================================
# 6. Example Usage
# =============================================
if __name__ == "__main__":
    user_query = "Explain Rule 'BILL AMOUNT SHOULD LESS THAN OTHER DOCUMENTS'?"
    run_query(user_query)
