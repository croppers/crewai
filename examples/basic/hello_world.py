"""
Basic example of creating and using a CrewAI agent with Ollama.
This example demonstrates the simplest way to create an agent and execute a task.
"""

from crewai import Agent, Crew, Task
from langchain_community.llms import Ollama

def main():
    # Initialize Ollama
    llm = Ollama(model="ollama/gemma:2b")
    
    # Create a basic agent
    agent = Agent(
        role='Researcher',
        goal='Research and analyze topics effectively',
        backstory='Expert researcher with years of experience in technology and AI',
        llm=llm,
        verbose=True
    )
    
    # Create a simple task
    task = Task(
        description="Research the impact of AI on healthcare",
        agent=agent,
        expected_output="A concise summary of the impact of AI on healthcare, highlighting key areas and potential future trends."
    )
    
    # Create a crew with a single agent and single task
    crew = Crew(
        agents=[agent],
        tasks=[task],
        verbose=True
    )
    
    # Execute the crew's task
    result = crew.kickoff()
    print("\nTask Result:")
    print(result)

if __name__ == "__main__":
    main() 