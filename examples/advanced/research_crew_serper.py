import os
from crewai import Agent, Crew, Task
from langchain_ollama import OllamaLLM
from crewai_tools import SerperDevTool
from dotenv import load_dotenv
from datetime import datetime, timedelta

# Load environment variables (including SERPER_API_KEY) from a .env file
load_dotenv()

# (In production, set SERPER_API_KEY via an environment variable.)
# os.environ["SERPER_API_KEY"] = "4b5ed8996b328949c9f1148ee58c2f3cafdda998"

# Debug wrapper for SerperDevTool
class DebugSerperDevTool(SerperDevTool):
    def run(self, *args, **kwargs):
        result = super().run(*args, **kwargs)
        print("\n[DEBUG] Raw SerperDevTool output:\n", result, "\n")
        with open("serper_raw_output.txt", "w") as f:
            f.write(str(result))
        return result

# Initialize the SerperDevTool (using the provided API key)
search_tool = DebugSerperDevTool(n_results=3)

# Initialize Ollama (using gemma:2b) for summarization
llm = OllamaLLM(model="gemma:2b", temperature=0.3, top_p=0.9)

# Calculate date range for recent articles (last 6 months)
six_months_ago = (datetime.now() - timedelta(days=180)).strftime("%Y-%m-%d")

# Create an agent (a "Researcher") that uses the SerperDevTool to search the internet.
researcher = Agent(
    role="Researcher",
    goal="Find and summarize the latest information on AI in healthcare.",
    backstory="An expert researcher who uses internet search tools to gather and summarize cutting-edge information.",
    tools=[search_tool],
    llm=llm,
    verbose=True
)

# Define a task that instructs the agent to search (using SerperDevTool) and then summarize (using Ollama).
search_and_summarize_task = Task(
    description=(
        "Use the SerperDevTool to search the internet for the query 'AI in healthcare' (with n_results=3). "
        "First, list the titles, URLs, and snippets of the top 3 results. "
        "Then, summarize these results into a concise report."
    ),
    expected_output=(
        "A list of the top 3 search results (title, URL, snippet), followed by a summary report."
    ),
    agent=researcher
)

# Assemble a crew (with the researcher agent and the search task) and execute it.
crew = Crew(
    agents=[researcher],
    tasks=[search_and_summarize_task],
    verbose=True
)

# Directly call the tool and print its output for verification
if __name__ == "__main__":
    print("\n[DIRECT TEST] Raw SerperDevTool output for 'AI in healthcare':")
    print(search_tool.run(search_query="AI in healthcare"))

    result = crew.kickoff()
    print("\nSearch and Summarize Report (using SerperDevTool and Ollama gemma:2b):")
    print(result) 