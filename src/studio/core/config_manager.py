import yaml
import os
from pathlib import Path
from typing import Dict, Any

class ConfigManager:
    def __init__(self, config_path: str = None):
        if config_path is None:
            # Default to src/studio/config/studio_config.yaml relative to this file
            # This file is in src/studio/core/
            base_dir = Path(__file__).resolve().parent.parent
            config_path = base_dir / "config" / "studio_config.yaml"
            
        self.config_path = Path(config_path)
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        if not self.config_path.exists():
            return {}
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def get_db_config(self, db_name: str) -> Dict[str, Any]:
        return self.config.get('databases', {}).get(db_name, {})
    
    def get_rag_config(self) -> Dict[str, Any]:
        return self.config.get('rag', {})
    
    def get_agent_config(self) -> Dict[str, Any]:
        return self.config.get('agents', {})
    
    def get_ui_config(self) -> Dict[str, Any]:
        return self.config.get('ui', {})
