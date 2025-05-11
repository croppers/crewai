"""
Example of creating a custom agent with custom tools.
This demonstrates how to extend agent capabilities with custom tools.
"""

from crewai import Agent, Task, Crew
# from crewai_tools import BaseTool # Not strictly needed if only using @tool decorator
from langchain_community.llms import Ollama
# from langchain.tools import Tool # We will use crewai's @tool instead
from crewai.tools import tool # Correct import for @tool
import requests
from bs4 import BeautifulSoup
from typing import List, Dict, Any
import json

@tool("WebSearchTool")
def search_web(query: str) -> str:
    """Search the web for information about a topic.Input is the search query.
    """
    try:
        # This is a placeholder implementation
        # In a real application, you would use a proper search API
        return f"Search results for: {query}\n" + \
               "1. Example result 1\n" + \
               "2. Example result 2\n" + \
               "3. Example result 3"
    except Exception as e:
        return f"Error performing web search: {str(e)}"

@tool("DataAnalysisTool")
def analyze_data(data: str, metrics: str) -> str: # Simplified for example, adjust as needed
    """Analyze data and generate insights based on specified metrics. Input is a JSON string of data and a comma-separated string of metrics.
    """
    try:
        # This is a placeholder implementation
        # In a real application, you would parse the data string (e.g., json.loads(data))
        # and the metrics string (e.g., metrics.split(','))
        analysis = {
            "parsed_data": data, # Placeholder
            "parsed_metrics": metrics, # Placeholder
            "summary": "Sample analysis summary",
            "recommendations": ["Sample recommendation 1", "Sample recommendation 2"]
        }
        return json.dumps(analysis, indent=2)
    except Exception as e:
        return f"Error analyzing data: {str(e)}"

def main():
    # Initialize Ollama
    llm = Ollama(model="ollama/gemma:2b")
    
    # Create custom tools
    # The tools are now automatically registered by the @tool decorator
    # and can be directly assigned to the agent if they are in the same scope
    # or imported if defined elsewhere.
    # For simplicity, we'll rely on them being in scope or explicitly pass them if needed.
    # CrewAI agents will discover tools available in the current context if not explicitly passed.
    # tools = [search_web, analyze_data] # This is how you'd typically pass them if needed by older versions or for clarity
    
    # Create agent with custom tools
    agent = Agent(
        role='Research Analyst',
        goal='Analyze and research topics thoroughly using available tools',
        backstory='Expert analyst with strong research and data analysis skills',
        llm=llm,
        tools=[search_web, analyze_data], # Pass the functions directly
        verbose=True
    )
    
    # Create and execute tasks
    research_task = Task(
        description="Research the latest developments in quantum computing and analyze the findings",
        agent=agent,
        expected_output="A comprehensive report on the latest developments in quantum computing, including an analysis of the findings."
    )
    
    # Create a crew to execute the task
    crew = Crew(
        agents=[agent],
        tasks=[research_task],
        verbose=True
    )
    
    result = crew.kickoff()
    
    print("\nTask Result:")
    print(result)

if __name__ == "__main__":
    main() 