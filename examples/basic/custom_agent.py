from crewai import Agent, Crew, Task
from langchain_community.llms import Ollama
from crewai.tools import tool
import requests
from bs4 import BeautifulSoup
from typing import List, Dict, Any
import json

@tool("WebSearchTool")
def search_web(query: str) -> str:
    """
    Search the web for the given query and return the first result.
    """
    url = f"https://www.google.com/search?q={query}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"}
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')
    result = soup.find('h3').text
    return result

llm = Ollama(model="ollama/gemma:2b")

# Create some agents...
researcher_agent = Agent(
    role='Researcher',
    goal='Research and analyze topics effectively',
    backstory='You are a researcher. You will be given a topic to research and analyze. You will provide a summary of your findings. You are a very informal researcher and you like to explain things in a down-to-earth way! So, be relaxed and chill with your explanations.',
    llm=llm,
    tools=[search_web],
    verbose=True,
)

# Create the tasks...
research_task = Task(
    description="Research the latest developments in quantum computing and analyze the findings.",
    agent=researcher_agent,
    expected_output="A comprehensive report on the latest quantum computing developments.",
)


# Create the crew...
crew = Crew(
    agents=[researcher_agent],
    tasks=[research_task],
    verbose=True,
)

# Kick off the crew...
result = crew.kickoff()
print(result)