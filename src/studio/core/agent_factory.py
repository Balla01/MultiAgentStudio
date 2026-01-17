import importlib
import os
import sys
from pathlib import Path
from typing import Dict, List, Any
import logging
from crewai import Crew, Agent, Task, Process


logger = logging.getLogger(__name__)

class AgentFactory:
    def __init__(self, crews_directory: str = "src/studio/crews"):
        self.crews_dir = Path(crews_directory).resolve()
        # Add parent of crews dir to path so we can import dynamically
        if str(self.crews_dir.parent) not in sys.path:
            sys.path.append(str(self.crews_dir.parent))

    def list_available_crews(self) -> List[str]:
        """Scan crews/ directory and return crew names"""
        crews = []
        if not self.crews_dir.exists():
            return []
            
        for crew_path in self.crews_dir.iterdir():
            if crew_path.is_dir() and (crew_path / "crew.py").exists():
                crews.append(crew_path.name)
        return crews

    def load_crew(self, crew_name: str, db_connections: Dict[str, Any]) -> Any:
        """
        Load a specific crew and inject database tools.
        """
        try:
            # Dynamic import: studio.crews.{crew_name}.crew
            # This assumes the directory structure allows this import.
            # If src/studio/crews is the path, and we run from root, 
            # we import studio.crews.name.crew
            
            module_name = f"studio.crews.{crew_name}.crew"
            module = importlib.import_module(module_name)
            
            # Convention: snake_case folder -> CamelCase class + "Crew"
            # e.g. 'test_crew' -> 'TestCrewCrew' (if strictly following cap), 
            # OR better: 'test_crew' -> 'TestCrew'
            
            # Let's convert snake_case to CamelCase
            camel_name = ''.join(x.title() for x in crew_name.split('_'))
            class_name = f"{camel_name}Crew"
            
            # Special case for manual override or simple names
            if not hasattr(module, class_name):
                # Try simple Capitalization
                simple_name = f"{crew_name.capitalize()}Crew"
                if hasattr(module, simple_name):
                    class_name = simple_name
                # Try without "Crew" suffix
                elif hasattr(module, camel_name):
                     class_name = camel_name
            
            if not hasattr(module, class_name):
                 # Fallback: try to find any class inheriting from BaseCrew or just 'Crew' in the file
                 # Or just expect strict naming convention
                 logger.error(f"Class {class_name} not found in {module_name}")
                 return None

            crew_class = getattr(module, class_name)
            
            # Create tools from DB connections
            # This requires the crew to accept 'tools' or 'db_tools' in init
            # Start with simple instantiation
            # If the Crew class follows CrewAI patterns, it might use decorators.
            # We assume here the user's Crew class has a structure that we can inject into.
            # If strictly CrewAI, we might need to inject tools into Agents.
            
            # For this studio, we assume the Crew class controls agent creation
            # and we might pass tools to it.
            
            # Placeholder for tool creation wrapper
            tools = self._create_db_tools(db_connections)
            
            # Instantiate
            # Assuming the Crew class accepts 'tools' argument or we verify how to inject
            try:
                crew_instance = crew_class(tools=tools)
            except TypeError:
                # Fallback if no tools arg
                crew_instance = crew_class()
                
            return crew_instance

        except Exception as e:
            logger.error(f"Failed to load crew {crew_name}: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _create_db_tools(self, db_connections: Dict) -> List:
        """Create CrewAI tools for each database"""
        tools = []
        # TODO: Wrap DB providers into LangChain/CrewAI tools
        # For now return raw list or empty if dependencies missing
        return tools
