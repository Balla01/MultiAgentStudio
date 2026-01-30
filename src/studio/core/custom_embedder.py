"""
Custom embedder wrapper for Alibaba GTE model to work with CrewAI memory system.
CrewAI expects embedders to follow a specific interface.
"""

from typing import List, Any
from sentence_transformers import SentenceTransformer
import numpy as np
from chromadb import Documents, Embeddings

# Import CrewAI's specific CustomEmbeddingFunction to satisfy Pydantic validation
try:
    from crewai.rag.embeddings.providers.custom.embedding_callable import CustomEmbeddingFunction
except ImportError:
    # Fallback/Mock for development environment if crewai internal paths change
    from chromadb import EmbeddingFunction
    class CustomEmbeddingFunction(EmbeddingFunction): pass

class AlibabGTEEmbedder(CustomEmbeddingFunction):
    """
    Custom embedder that wraps your existing Alibaba-NLP/gte-large-en-v1.5 model
    to be compatible with CrewAI's memory system.
    """
    
    def __init__(self, model_name: str = "Alibaba-NLP/gte-large-en-v1.5", **kwargs):
        """
        Initialize the Alibaba GTE embedding model.
        
        Args:
            model_name: The Hugging Face model identifier
        """
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, trust_remote_code=True)
        self.dimension = self.model.get_sentence_embedding_dimension()
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of documents.
        """
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False
        )
        return embeddings.tolist()
    
    def embed_query(self, text: str = None, input: Any = None) -> List[float]:
        """
        Embed a single query text.
        Supports both 'text' positional/legacy and 'input' kwarg for ChromaDB 0.4+.
        """
        target = text or input
        if not target:
            return []
            
        embedding = self.model.encode(
            target,
            convert_to_numpy=True,
            show_progress_bar=False
        )
        return embedding.tolist()
    
    def __call__(self, input: Documents) -> Embeddings:
        """
        ChromaDB compatible call method.
        """
        return self.embed_documents(input)

    def name(self) -> str:
        """
        Return the name of the embedding function.
        Required by recent ChromaDB versions for validation.
        """
        return self.model_name


class CrewAIEmbedderAdapter:
    """
    Adapter to make AlibabGTEEmbedder compatible with CrewAI's expected interface.
    This handles the specific format CrewAI expects for embedder configurations.
    """
    
    def __init__(self, model_name: str = "Alibaba-NLP/gte-large-en-v1.5"):
        self.embedder = AlibabGTEEmbedder(model_name)
        self.provider = "custom_alibaba_gte"
        
    def get_config(self) -> dict:
        """
        Returns embedder configuration in CrewAI-compatible format.
        
        Note: CrewAI typically expects 'provider' and 'config' keys,
        but for custom embedders, we return the embedder instance directly.
        """
        return {
            "provider": "custom",
            "config": {
                "model_name": self.embedder.model_name,
                "dimension": self.embedder.dimension,
                "embedding_callable": AlibabGTEEmbedder
            },
            "embedding_callable": AlibabGTEEmbedder
        }
