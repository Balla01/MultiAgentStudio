"""
Central memory configuration for CrewAI integration.
Handles memory settings, storage paths, and custom embedder setup.
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any
from .custom_embedder import CrewAIEmbedderAdapter

class MemoryConfigManager:
    """
    Manages memory configuration for CrewAI crews with custom embeddings.
    """
    
    def __init__(
        self,
        storage_dir: Optional[str] = None,
        embedding_model: str = "Alibaba-NLP/gte-large-en-v1.5",
        enable_short_term: bool = True,
        enable_long_term: bool = True,
        enable_entity: bool = True
    ):
        """
        Initialize memory configuration.
        
        Args:
            storage_dir: Custom storage directory (default: ./storage/memory)
            embedding_model: Embedding model name
            enable_short_term: Enable short-term memory
            enable_long_term: Enable long-term memory
            enable_entity: Enable entity memory
        """
        # Set storage directory
        if storage_dir:
            self.storage_dir = Path(storage_dir)
        else:
            self.storage_dir = Path("./storage/memory")
        
        # Create storage directory
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Set environment variable for CrewAI
        os.environ["CREWAI_STORAGE_DIR"] = str(self.storage_dir.absolute())
        
        # Initialize custom embedder
        self.embedder_adapter = CrewAIEmbedderAdapter(embedding_model)
        
        # Memory settings
        self.enable_short_term = enable_short_term
        self.enable_long_term = enable_long_term
        self.enable_entity = enable_entity
        
        print(f"✅ Memory storage configured at: {self.storage_dir.absolute()}")
        print(f"✅ Using embedding model: {embedding_model}")
    
    def get_memory_config(self) -> Dict[str, Any]:
        """
        Get memory configuration for CrewAI Crew initialization.
        
        Returns:
            Dictionary with memory settings
        """
        return {
            "memory": True,  # Enable memory system
            "verbose": True,  # Show memory operations
            # Note: Custom embedder is set separately in crew initialization
        }
    
    def get_embedder_instance(self):
        """
        Get the embedder instance for direct use.
        
        Returns:
            AlibabGTEEmbedder instance
        """
        return self.embedder_adapter.embedder
    
    def get_storage_path(self, memory_type: str = None) -> Path:
        """
        Get path for specific memory type storage.
        
        Args:
            memory_type: 'short_term', 'long_term', 'entity', 'knowledge', or None for base
            
        Returns:
            Path to storage directory
        """
        if memory_type:
            return self.storage_dir / memory_type
        return self.storage_dir
    
    def reset_memory(self, memory_type: str = 'all'):
        """
        Reset memory storage.
        
        Args:
            memory_type: 'short', 'long', 'entity', 'knowledge', or 'all'
        """
        import shutil
        
        if memory_type == 'all':
            if self.storage_dir.exists():
                shutil.rmtree(self.storage_dir)
                self.storage_dir.mkdir(parents=True, exist_ok=True)
                print(f"✅ All memory reset: {self.storage_dir}")
        else:
            type_map = {
                'short': 'short_term_memory',
                'long': 'long_term_memory',
                'entity': 'entities',
                'knowledge': 'knowledge'
            }
            
            if memory_type in type_map:
                mem_path = self.storage_dir / type_map[memory_type]
                if mem_path.exists():
                    shutil.rmtree(mem_path)
                    print(f"✅ {memory_type} memory reset: {mem_path}")
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get statistics about memory storage.
        
        Returns:
            Dictionary with memory storage information
        """
        import os
        
        stats = {
            "storage_dir": str(self.storage_dir.absolute()),
            "exists": self.storage_dir.exists(),
            "memory_types": {}
        }
        
        if self.storage_dir.exists():
            for item in self.storage_dir.iterdir():
                if item.is_dir():
                    # Count files in directory
                    file_count = len(list(item.rglob("*")))
                    size = sum(f.stat().st_size for f in item.rglob("*") if f.is_file())
                    stats["memory_types"][item.name] = {
                        "files": file_count,
                        "size_mb": round(size / (1024 * 1024), 2)
                    }
                elif item.is_file():
                    size = item.stat().st_size
                    stats["memory_types"][item.name] = {
                        "size_mb": round(size / (1024 * 1024), 2)
                    }
        
        return stats
