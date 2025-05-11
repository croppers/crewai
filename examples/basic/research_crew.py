"""
Example of creating a research crew with multiple specialized agents.
This demonstrates a more complex multi-agent system with specialized roles.
"""

from crewai import Agent, Crew, Task
from langchain_community.llms import Ollama
from typing import Dict, List

def main():
    # Initialize Ollama
    llm = Ollama(model="ollama/gemma:2b")
    
    # Create specialized agents
    researcher = Agent(
        role='Researcher',
        goal='Research topics thoroughly and gather comprehensive information',
        backstory='Expert researcher with a strong background in technology and AI',
        llm=llm,
        verbose=True
    )
    
    analyst = Agent(
        role='Analyst',
        goal='Analyze research findings and identify key insights',
        backstory='Data analyst with expertise in technology trends and market analysis',
        llm=llm,
        verbose=True
    )
    
    writer = Agent(
        role='Writer',
        goal='Create engaging and informative content based on research and analysis',
        backstory='Experienced content writer specializing in technology and AI topics',
        llm=llm,
        verbose=True
    )
    
    editor = Agent(
        role='Editor',
        goal='Review and improve content for clarity, accuracy, and engagement',
        backstory='Senior editor with experience in technology publications',
        llm=llm,
        verbose=True
    )
    
    # Create tasks with dependencies
    research_task = Task(
        description="Research the latest developments in AI and their impact on healthcare, focusing on recent breakthroughs and future implications",
        agent=researcher,
        expected_output="A detailed report on AI developments in healthcare, including breakthroughs and future implications."
    )
    
    analysis_task = Task(
        description="Analyze the research findings, identify key trends, and evaluate the potential impact on healthcare",
        agent=analyst,
        expected_output="An analysis of research findings, identifying key trends and evaluating healthcare impact."
    )
    
    writing_task = Task(
        description="Write a comprehensive report about AI in healthcare, incorporating research findings and analysis",
        agent=writer,
        expected_output="A comprehensive and well-structured report on AI in healthcare."
    )
    
    editing_task = Task(
        description="Review and edit the report for clarity, accuracy, and engagement",
        agent=editor,
        expected_output="A polished, error-free, and engaging report on AI in healthcare."
    )
    
    # Create and run crew
    crew = Crew(
        agents=[researcher, analyst, writer, editor],
        tasks=[research_task, analysis_task, writing_task, editing_task],
        verbose=True
    )
    
    # Execute the crew's tasks
    result = crew.kickoff()
    
    print("\nCrew Execution Result:")
    print(result)

if __name__ == "__main__":
    main() 