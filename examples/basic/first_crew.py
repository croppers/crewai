"""
Example of creating a simple crew with multiple agents working together.
This demonstrates basic multi-agent collaboration using CrewAI and Ollama.
"""

from crewai import Agent, Crew, Task
from langchain_community.llms import Ollama

def main():
    # Initialize Ollama
    llm = Ollama(model="ollama/gemma:2b")
    
    # Create agents
    researcher = Agent(
        role='Researcher',
        goal='Research topics thoroughly and gather relevant information',
        backstory='Expert researcher with a strong background in technology and AI',
        llm=llm,
        verbose=True
    )
    
    writer = Agent(
        role='Writer',
        goal='Create engaging and informative content based on research',
        backstory='Experienced content writer specializing in technology and AI topics',
        llm=llm,
        verbose=True
    )
    
    # Create tasks
    research_task = Task(
        description="Research the latest developments in AI and their impact on healthcare",
        agent=researcher,
        expected_output="A detailed report on the latest AI developments in healthcare and their impact."
    )
    
    writing_task = Task(
        description="Write a comprehensive blog post about AI in healthcare, focusing on recent developments and future implications",
        agent=writer,
        expected_output="An engaging and informative blog post about AI in healthcare, approximately 500-700 words."
    )
    
    # Create and run crew
    crew = Crew(
        agents=[researcher, writer],
        tasks=[research_task, writing_task],
        verbose=True
    )
    
    # Execute the crew's tasks
    result = crew.kickoff()
    
    print("\nCrew Execution Result:")
    print(result)

if __name__ == "__main__":
    main() 