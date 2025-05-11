"""
Content Team Application
A complete application demonstrating a content creation team using CrewAI and Ollama.
"""

from crewai import Agent, Crew, Task, Process
from langchain_community.llms import Ollama
from typing import List, Dict, Any
import asyncio
import json
import os
import logging
from datetime import datetime
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ContentTeam:
    """A content creation team using CrewAI agents."""
    
    def __init__(self, config_path: str):
        self.config = self.load_config(config_path)
        self.setup_environment()
        self.setup_crew()
    
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from JSON file."""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading config: {str(e)}")
            raise
    
    def setup_environment(self):
        """Set up the working environment."""
        os.makedirs(self.config['output_dir'], exist_ok=True)
        self.setup_logging()
    
    def setup_logging(self):
        """Set up logging to file."""
        log_file = Path(self.config['output_dir']) / f"content_team_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        logger.addHandler(file_handler)
    
    def setup_crew(self):
        """Set up the content creation crew."""
        # Initialize Ollama
        llm = Ollama(model=self.config['model'])
        
        # Create agents
        self.agents = {}
        for agent_id, agent_config in self.config['agents'].items():
            self.agents[agent_id] = Agent(
                role=agent_config['role'],
                goal=agent_config['goal'],
                backstory=agent_config['backstory'],
                llm=llm,
                verbose=self.config['verbose']
            )
        
        # Create tasks
        self.tasks = [
            Task(
                description=task['description'],
                agent=self.agents[task['agent']]
            )
            for task in self.config['tasks']
        ]
        
        # Create crew
        self.crew = Crew(
            agents=list(self.agents.values()),
            tasks=self.tasks,
            process=Process[self.config['process']],
            verbose=self.config['verbose']
        )
    
    async def execute(self) -> Dict[str, Any]:
        """Execute the content creation process."""
        try:
            logger.info("Starting content creation process")
            start_time = datetime.now()
            
            # Execute crew tasks
            results = await self.crew.execute()
            
            # Save results
            self.save_results(results, start_time)
            
            return results
        except Exception as e:
            logger.error(f"Error in content creation: {str(e)}")
            raise
    
    def save_results(self, results: Dict[str, Any], start_time: datetime):
        """Save content creation results to file."""
        try:
            output_file = Path(self.config['output_dir']) / f"content_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            output_data = {
                "timestamp": datetime.now().isoformat(),
                "duration": (datetime.now() - start_time).total_seconds(),
                "config": self.config,
                "results": results
            }
            
            with open(output_file, 'w') as f:
                json.dump(output_data, f, indent=2)
            
            logger.info(f"Results saved to {output_file}")
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")

def main():
    """Run the content team application."""
    try:
        # Create content team
        team = ContentTeam("config.json")
        
        # Execute content creation
        results = asyncio.run(team.execute())
        
        # Print results
        print("\nContent Creation Results:")
        for agent_id, result in results.items():
            print(f"\n{team.agents[agent_id].role}:")
            print(result)
        
    except Exception as e:
        print(f"\nError in content creation: {str(e)}")
        logger.error(f"Application error: {str(e)}")

if __name__ == "__main__":
    main() 