from crewai import Agent, Task
from typing import List, Dict, Any, Optional

def get_agents(db_connections: Optional[Dict[str, Any]] = None) -> List[Agent]:
    """
    Create and return agents for this crew.
    """
    # Create tools from database connections
    tools = []
    # (Tools would be wrapped here)
    
    # Define agents
    researcher = Agent(
        role="Senior Research Analyst",
        goal="Uncover cutting-edge developments in AI and data science",
        backstory="""You are an expert research analyst with a keen eye for detail.
        You excel at finding and synthesizing information from various sources.
        You have access to long-term memory to build on previous research.""",
        tools=tools,
        verbose=True,
        allow_delegation=False
    )
    
    writer = Agent(
        role="Tech Content Strategist",
        goal="Craft compelling content on tech advancements",
        backstory="""You are a skilled writer who can translate complex technical
        concepts into engaging content. You remember writing styles and topics
        from past projects through your memory system.""",
        tools=tools,
        verbose=True,
        allow_delegation=False
    )
    
    return [researcher, writer]

def get_tasks(agents: List[Agent]) -> List[Task]:
    """
    Create and return tasks for this crew.
    """
    if len(agents) < 2:
        return []
        
    researcher, writer = agents
    
    research_task = Task(
        description="""Conduct comprehensive research on the user's query.
        Use your memory to build on previous research and avoid duplication.
        Focus on recent developments and breakthrough innovations.""",
        expected_output="""A detailed research report with:
        - Key findings and innovations
        - Relevant sources and references
        - Connections to previous research (from memory)""",
        agent=researcher
    )
    
    writing_task = Task(
        description="""Create engaging content based on the research.
        Reference your memory of past writing projects to maintain consistency.
        Ensure the content is accessible yet informative.""",
        expected_output="""A well-structured article that:
        - Explains complex concepts clearly
        - Engages the target audience
        - Builds on previous content themes (from memory)""",
        agent=writer,
        context=[research_task]
    )
    
    return [research_task, writing_task]
