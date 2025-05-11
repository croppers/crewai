"""
Advanced example of creating custom tools for CrewAI agents.
This demonstrates how to create sophisticated tools with proper error handling and type hints.
"""

from typing import List, Dict, Any, Optional
from crewai.tools import tool # Import the @tool decorator
from pydantic import BaseModel, Field # Still can be used for input validation if desired, or simplify inputs
import requests
from bs4 import BeautifulSoup
import json
import logging
from datetime import datetime
import pandas as pd
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@tool("WebScraperTool")
def scrape_website(url: str, selectors: str) -> str:
    """Scrape content from a website. 
    Input must be a URL and a comma-separated string of CSS selectors.
    Example: scrape_website(url="https://example.com", selectors="h1,p.some_class")
    """
    try:
        logger.info(f"Scraping URL: {url} with selectors: {selectors}")
        selector_list = [s.strip() for s in selectors.split(',')]
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        results = {}
        
        for selector in selector_list:
            elements = soup.select(selector)
            results[selector] = [elem.text.strip() for elem in elements]
        
        return json.dumps(results, indent=2)
    except requests.RequestException as e:
        logger.error(f"Request error: {str(e)}")
        return f"Error scraping website: {str(e)}"
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return f"Error scraping website: {str(e)}"

@tool("DataAnalysisTool")
def analyze_data_func(data_json: str, metrics_str: str, group_by_col: Optional[str] = None) -> str:
    """Analyze data (provided as a JSON string of records) and generate insights.
    Input: 
        data_json: A JSON string representing a list of dictionaries (records).
        metrics_str: A comma-separated string of metrics to calculate (e.g., 'mean,median,std').
        group_by_col: Optional column name to group by.
    Example: analyze_data_func(data_json='[{"colA": 1, "colB": 10}, {"colA": 2, "colB": 20}]', metrics_str='mean,std', group_by_col='colA')
    """
    try:
        logger.info("Starting data analysis func")
        data_list = json.loads(data_json)
        df = pd.DataFrame(data_list)
        metric_list = [m.strip() for m in metrics_str.split(',')]
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "metrics": {},
            "summary": {},
            "recommendations": []
        }
        
        numeric_df = df.select_dtypes(include=np.number) # Select only numeric columns for pandas operations

        for metric in metric_list:
            if not numeric_df.empty:
                if metric == "mean":
                    results["metrics"]["mean"] = numeric_df.mean().to_dict()
                elif metric == "median":
                    results["metrics"]["median"] = numeric_df.median().to_dict()
                elif metric == "std":
                    results["metrics"]["std"] = numeric_df.std().to_dict()
            else:
                 results["metrics"][metric] = "No numeric data to calculate " + metric

        if group_by_col and group_by_col in df.columns:
            grouped = df.groupby(group_by_col)
            numeric_grouped_means = df.groupby(group_by_col).mean(numeric_only=True) # Ensure numeric_only for mean
            results["group_analysis"] = {
                "counts": grouped.size().to_dict(),
                "means": numeric_grouped_means.to_dict() if not numeric_grouped_means.empty else "No numeric data for grouped means"
            }
        
        results["summary"] = {
            "total_records": len(df),
            "columns": list(df.columns),
            "missing_values": df.isnull().sum().to_dict()
        }
        
        if "missing_values" in results["summary"]:
            missing = results["summary"]["missing_values"]
            for col, count in missing.items():
                if count > 0:
                    results["recommendations"].append(f"Consider handling missing values in column: {col}")
        
        return json.dumps(results, indent=2, default=str) # Add default=str for non-serializable numpy types
    except Exception as e:
        logger.error(f"Error in data analysis: {str(e)}")
        return f"Error analyzing data: {str(e)}"

@tool("MarketResearchTool")
def research_market_func(industry: str, metrics_str: str, timeframe: str = "1y") -> str:
    """Conduct market research for a specific industry.
    Input:
        industry: The industry to research.
        metrics_str: A comma-separated string of market metrics to analyze (e.g., 'market_size,growth_rate,key_players').
        timeframe: Analysis timeframe (e.g., '1y', '6m', 'Q1 2023').
    Example: research_market_func(industry="AI in Healthcare", metrics_str="market_size,trends", timeframe="2023")
    """
    try:
        logger.info(f"Conducting market research for {industry} with metrics: {metrics_str}")
        metric_list = [m.strip() for m in metrics_str.split(',')]
        results = {
            "industry": industry,
            "timeframe": timeframe,
            "metrics": {metric: "sample value for " + metric for metric in metric_list},
            "trends": ["Sample trend 1 for " + industry, "Sample trend 2", "Sample trend 3"],
            "recommendations": ["Sample recommendation 1 for " + industry, "Sample recommendation 2"]
        }
        return json.dumps(results, indent=2)
    except Exception as e:
        logger.error(f"Error in market research: {str(e)}")
        return f"Error conducting market research: {str(e)}"

def main():
    """Example usage of the custom tools."""
    from crewai import Agent, Task, Crew
    from langchain_ollama import OllamaLLM

    # Tools are now functions decorated with @tool
    tools = [scrape_website, analyze_data_func, research_market_func]
    
    # Initialize Ollama LLM
    llm = OllamaLLM(model="ollama/gemma:2b")

    # Create agent with custom tools
    agent = Agent(
        role='Market Research Analyst',
        goal='Conduct comprehensive market research and analysis using available tools',
        backstory='Expert market analyst proficient in web scraping, data analysis, and market trend identification.',
        llm=llm,
        tools=tools,
        verbose=True,
        allow_delegation=False
    )
    
    # Define the task description string first
    task_description_str = (
        "Research the AI market in healthcare. Scrape relevant information from 'https://example.com/ai-healthcare-news' using selectors 'h2.article-title,p.summary'. " + 
        "Then, analyze the collected data (imagine the scraped text is '[{{\"scraped_title\": \"AI Breakthrough in Cancer Detection\", \"summary\": \"New AI model shows promise...\"}}]') focusing on metrics 'sentiment,key_entities'. " + 
        "Finally, provide a market overview for 'AI in Healthcare' for '2024' focusing on 'market_size,main_trends'."
    )

    # Create a task for the agent
    research_task = Task(
        description=task_description_str, # Use the pre-defined variable
        agent=agent,
        expected_output="A comprehensive market research report on AI in healthcare, including a summary of scraped news, data analysis of sentiment and key entities, and an overview of market size and main trends for 2024.",
        tools=tools 
    )
    
    # Create a crew to execute the task
    crew = Crew(
        agents=[agent],
        tasks=[research_task],
        verbose=True
    )
    
    # Execute the crew's task
    result = crew.kickoff()
    
    print("\nTask Result:")
    print(result)

if __name__ == "__main__":
    main()  