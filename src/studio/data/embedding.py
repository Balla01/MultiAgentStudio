from typing import List, Union
import logging

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    logging.warning("SentenceTransformer not found.")

class EmbeddingGenerator:
    def __init__(self, model_name: str = "Alibaba-NLP/gte-large-en-v1.5"):
        self.model_name = model_name
        self.model = None
        self._load_model()

    def _load_model(self):
        try:
            self.model = SentenceTransformer(self.model_name, trust_remote_code=True)
        except Exception as e:
            logging.error(f"Failed to load embedding model {self.model_name}: {e}")

    def generate(self, texts: Union[str, List[str]]) -> List[List[float]]:
        if not self.model:
            return []
        
        if isinstance(texts, str):
            texts = [texts]
            
        return self.model.encode(texts).tolist()
