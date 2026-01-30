"""
Script to seed memory with a few sample exchanges and answer live queries.

Supports two memory techniques:
1. ConversationBufferWindowMemory - keeps last k conversations in a sliding window
2. VectorStoreRetrieverMemory - uses vector similarity to retrieve relevant past conversations
"""
import json
import os
from typing import List, Dict, Tuple, Optional

from crewai import Agent, Task, Crew, Process, LLM
from langchain.memory import ConversationBufferWindowMemory, VectorStoreRetrieverMemory, CombinedMemory
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.docstore import InMemoryDocstore
from langchain.embeddings.base import Embeddings
from sentence_transformers import SentenceTransformer
import faiss

# Configuration
MEMORY_APPROACH = 1  # 1: CombinedMemory (Vector + Window), 2: ConversationBufferWindowMemory only
EMBEDDING_MODEL_NAME = "Alibaba-NLP/gte-large-en-v1.5"
VECTOR_RETRIEVER_K = 3  # Number of relevant past conversations to retrieve (Approach 1 only)
WINDOW_MEMORY_K = 2  # Number of most recent conversations for window memory (Approach 1 only)

MAX_VECTORSTORE_ITEMS = 8  # Limit for FAISS memory (most recent only)


BUFFER_WINDOW_SIZE = 10  # Number of conversations to keep in buffer (Approach 2 only)
MISTRAL_MODEL = "mistral/mistral-large-latest"
# MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "")
MISTRAL_API_KEY = '0TD9nsBifR6Lkr1kOag9aikbCBImYfGg'#os.getenv("MISTRAL_API_KEY")

MEMORY_STORAGE_DIR = "/home/ntlpt19/personal_projects/MultiAgentStudio/storage/memory"
EXCHANGES_PATH = os.path.join(MEMORY_STORAGE_DIR, "exchanges.json")
FAISS_STORAGE_DIR = os.path.join(MEMORY_STORAGE_DIR, "faiss")


class SentenceTransformerEmbeddings(Embeddings):
    """Custom embeddings class using SentenceTransformer."""

    def __init__(self, model_name: str):
        """Initialize the embedding model."""
        self.model = SentenceTransformer(
            model_name,
            trust_remote_code=True,
            device="cpu"
        )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents."""
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query text."""
        embedding = self.model.encode([text], convert_to_numpy=True)
        return embedding[0].tolist()


def get_embedding_model():
    """Get the embedding model instance."""
    return SentenceTransformerEmbeddings(EMBEDDING_MODEL_NAME)


def get_llm():
    """Get the LLM instance based on configuration."""
    if not MISTRAL_API_KEY:
        print("⚠ WARNING: MISTRAL_API_KEY is not set. Set it to run queries.")
    return LLM(model=MISTRAL_MODEL, api_key=MISTRAL_API_KEY)


def ensure_storage_dir():
    """Ensure memory storage directory exists."""
    os.makedirs(MEMORY_STORAGE_DIR, exist_ok=True)


def load_exchanges() -> List[Dict[str, str]]:
    """Load persisted exchanges from disk."""
    if not os.path.exists(EXCHANGES_PATH):
        return []
    with open(EXCHANGES_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        return []
    return [item for item in data if "input" in item and "output" in item]


def save_exchanges(exchanges: List[Dict[str, str]]) -> None:
    """Persist exchanges to disk."""
    ensure_storage_dir()
    with open(EXCHANGES_PATH, "w", encoding="utf-8") as f:
        json.dump(exchanges, f, ensure_ascii=True, indent=2)


def load_or_create_vectorstore(embeddings) -> FAISS:
    """Load FAISS index from disk or create a new one."""
    if os.path.exists(os.path.join(FAISS_STORAGE_DIR, "index.faiss")):
        print(f"Loading FAISS index from: {FAISS_STORAGE_DIR}")
        return FAISS.load_local(
            FAISS_STORAGE_DIR,
            embeddings,
            allow_dangerous_deserialization=True
        )

    # Setup the Vector Database
    embedding_size = 1024  # Dimensions for Alibaba-NLP/gte-large-en-v1.5
    index = faiss.IndexFlatL2(embedding_size)

    # Create FAISS vectorstore
    return FAISS(
        embedding_function=embeddings.embed_query,
        index=index,
        docstore=InMemoryDocstore({}),
        index_to_docstore_id={}
    )


def setup_retriever() -> Tuple[object, FAISS, Embeddings]:
    """Setup FAISS retriever with SentenceTransformer embeddings."""
    print(f"Setting up FAISS retriever with embedding model: {EMBEDDING_MODEL_NAME}")

    # Get the embedding model
    embeddings = get_embedding_model()

    vectorstore = load_or_create_vectorstore(embeddings)

    # Setup the Retriever
    retriever = vectorstore.as_retriever(search_kwargs=dict(k=VECTOR_RETRIEVER_K))

    print(f"FAISS retriever initialized (k={VECTOR_RETRIEVER_K})")
    return retriever, vectorstore, embeddings


def setup_combined_memory(retriever):
    """Setup CombinedMemory with both vector and window memory."""
    print("\n" + "="*80)
    print("Setting up COMBINED MEMORY")
    print("="*80)

    # 1. Setup Vector Memory (For long-term / topic-based retrieval)
    vector_memory = VectorStoreRetrieverMemory(
        retriever=retriever,
        memory_key="relevant_history"
    )
    print(f"✓ Vector Memory: retrieves top {VECTOR_RETRIEVER_K} relevant past conversations")

    # 2. Setup Window Memory (For immediate context like "it" or "that")
    window_memory = ConversationBufferWindowMemory(
        k=WINDOW_MEMORY_K,
        memory_key="recent_history",
        input_key="input"
    )
    print(f"✓ Window Memory: keeps last {WINDOW_MEMORY_K} conversation(s)")

    # 3. Combine them
    memory = CombinedMemory(memories=[vector_memory, window_memory])
    print("✓ Combined Memory initialized successfully!")

    return memory


def build_prompt():
    """Build the prompt template based on memory approach."""
    if MEMORY_APPROACH == 1:
        custom_template = """You are an AI assistant answering regulatory and banking-related questions.

            You are given:
            1. Relevant pieces of past conversation retrieved by semantic similarity (long-term memory).
            2. The most recent interaction (short-term context).
            3. The current user question.

            INSTRUCTIONS:
            - Carefully check whether the retrieved past conversation is relevant to the current question.
            - Use the retrieved past conversation ONLY if it directly helps answer the current question.
            - If the retrieved history is partially relevant, use only the relevant parts.
            - Do NOT invent details that are not present in the retrieved history.
            - If the answer cannot be fully derived from the retrieved history, answer using general domain knowledge and clearly explain your reasoning.
            - Give priority to factual accuracy and regulatory correctness.

            Relevant past conversation:
            {relevant_history}

            Recent interaction:
            {recent_history}

            Current question:
            {input}

            Answer the question clearly and concisely.
            """
        return PromptTemplate(
            input_variables=["relevant_history", "recent_history", "input"],
            template=custom_template
        )

    custom_template = """You are an AI assistant answering regulatory and banking-related questions.

        You are given:
        1. Past conversation history (last {buffer_size} conversations).
        2. The current user question.

        INSTRUCTIONS:
        - Carefully check whether the conversation history is relevant to the current question.
        - Use the conversation history ONLY if it directly helps answer the current question.
        - If the history is partially relevant, use only the relevant parts.
        - Do NOT invent details that are not present in the history.
        - If the answer cannot be fully derived from the history, answer using general domain knowledge and clearly explain your reasoning.
        - Give priority to factual accuracy and regulatory correctness.

        Conversation history:
        {{history}}

        Current question:
        {{input}}

        Answer the question clearly and concisely.
        """.format(buffer_size=BUFFER_WINDOW_SIZE)

    return PromptTemplate(
        input_variables=["history", "input"],
        template=custom_template
    )


def seed_memory_exchanges(memory) -> List[Dict[str, str]]:
    """Seed memory with persisted or sample exchanges."""
    exchanges = load_exchanges()
    if len(exchanges)> MAX_VECTORSTORE_ITEMS:
        exchanges = exchanges[(len(exchanges) - MAX_VECTORSTORE_ITEMS):]
        
    # if exchanges:
    #     print(f"Loaded {len(exchanges)} exchange(s) from disk.")
    # else:
    #     exchanges = [
    #         {
    #             "input": "What is CRR and why do banks maintain it?",
    #             "output": "CRR is the Cash Reserve Ratio, a share of deposits banks keep with the central bank to ensure liquidity and stability."
    #         }
    #     ]
    #     save_exchanges(exchanges)

    for idx, exchange in enumerate(exchanges, 1):
        memory.save_context({"input": exchange["input"]}, {"output": exchange["output"]})
        print(f"Seeded exchange {idx}")

    return exchanges


def build_vector_texts(exchanges: List[Dict[str, str]]) -> List[str]:
    """Build vectorstore texts from exchanges."""
    texts = []
    for exchange in exchanges:
        texts.append(f"User: {exchange['input']}\nAssistant: {exchange['output']}")
    return texts


def enforce_vectorstore_limit(
    retriever,
    embeddings: Embeddings,
    exchanges: List[Dict[str, str]]
) -> Optional[FAISS]:
    """Rebuild vectorstore with only the most recent exchanges."""
    if retriever is None or embeddings is None:
        return None

    limited = exchanges[-MAX_VECTORSTORE_ITEMS:]
    texts = build_vector_texts(limited)
    vectorstore = FAISS.from_texts(texts, embeddings)
    retriever.vectorstore = vectorstore
    ensure_storage_dir()
    vectorstore.save_local(FAISS_STORAGE_DIR)
    return vectorstore


def generate_response(
    llm: LLM,
    memory,
    prompt: PromptTemplate,
    user_query: str,
    exchanges: List[Dict[str, str]],
    retriever=None,
    embeddings: Optional[Embeddings] = None
) -> Tuple[str, Optional[FAISS]]:
    """Generate a response and update memory."""
    memory_vars = memory.load_memory_variables({"input": user_query})
    if MEMORY_APPROACH == 1:
        full_prompt = prompt.format(
            relevant_history=memory_vars.get("relevant_history", ""),
            recent_history=memory_vars.get("recent_history", ""),
            input=user_query
        )
    else:
        full_prompt = prompt.format(
            history=memory_vars.get("history", ""),
            input=user_query
        )

    response = llm.call(full_prompt)
    memory.save_context({"input": user_query}, {"output": response})
    exchanges.append({"input": user_query, "output": response})
    save_exchanges(exchanges)
    vectorstore = enforce_vectorstore_limit(retriever, embeddings, exchanges)
    return response, vectorstore


def smoke_test_llm(llm: LLM) -> None:
    """Quick LLM sanity check."""
    if not MISTRAL_API_KEY:
        print("Skipping LLM smoke test because MISTRAL_API_KEY is not set.")
        return
    response = llm.call("Explain vector databases in simple words.")
    print(response)

# Setup memory based on approach
if MEMORY_APPROACH == 1:
    print("\n" + "="*80)
    print("APPROACH 1: Using CombinedMemory (Vector + Window)")
    print("="*80)
    retriever, vectorstore, embeddings = setup_retriever()
    memory = setup_combined_memory(retriever)
else:
    print("\n" + "="*80)
    print(f"APPROACH 2: Using ConversationBufferWindowMemory only (k={BUFFER_WINDOW_SIZE})")
    print("="*80)
    memory = ConversationBufferWindowMemory(k=BUFFER_WINDOW_SIZE)
    retriever = None
    vectorstore = None
    embeddings = None
    print(f"✓ Buffer Window Memory initialized (k={BUFFER_WINDOW_SIZE})")

exchanges = seed_memory_exchanges(memory)
vectorstore = enforce_vectorstore_limit(retriever, embeddings, exchanges)
print("\nMemory seeded.")

prompt = build_prompt()

llm = get_llm()


def retrieve_history(user_query):
    response, vectorstore = generate_response(
        llm,
        memory,
        prompt,
        user_query,
        exchanges,
        retriever,
        embeddings
    )
    # print(response)
    print("\n" + "="*80)
    print("Memory Summary:")
    print("="*80)
    print(f"Approach: {MEMORY_APPROACH}")
    print(f"Memory type: {type(memory).__name__}")
    if MEMORY_APPROACH == 1:
        print(f"Components: Vector Memory (k={VECTOR_RETRIEVER_K}) + Window Memory (k={WINDOW_MEMORY_K})")
    else:
        print(f"Buffer window size: {BUFFER_WINDOW_SIZE}")
    return response

if __name__ == "__main__":
    user_query = 'where it can be applied ?'
    res = retrieve_history(user_query)
    print("="*30)
    print("="*30)
    print(res)
