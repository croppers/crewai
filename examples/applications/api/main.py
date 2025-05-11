"""
CrewAI API Application
A FastAPI application that provides an API for running CrewAI crews.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from crewai import Agent, Crew, Task
from langchain_community.llms import Ollama
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import uvicorn
import asyncio
import logging
import json
from datetime import datetime
from pathlib import Path
import os

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="CrewAI API",
    description="API for running CrewAI crews with Ollama",
    version="1.0.0"
)

# Models
class AgentConfig(BaseModel):
    role: str = Field(..., description="Agent role")
    goal: str = Field(..., description="Agent goal")
    backstory: str = Field(..., description="Agent backstory")

class TaskConfig(BaseModel):
    description: str = Field(..., description="Task description")
    agent_index: int = Field(..., description="Index of the agent to execute the task")

class CrewRequest(BaseModel):
    topic: str = Field(..., description="Topic for the crew to work on")
    agents: List[AgentConfig] = Field(..., description="List of agents")
    tasks: List[TaskConfig] = Field(..., description="List of tasks")
    model: str = Field("gemma:2b", description="Ollama model to use")
    output_dir: str = Field("api_output", description="Output directory for results")

class CrewResponse(BaseModel):
    crew_id: str
    status: str
    message: str

# Global state
active_crews: Dict[str, Dict[str, Any]] = {}

def save_results(crew_id: str, results: Dict[str, Any], config: Dict[str, Any]):
    """Save crew execution results to file."""
    try:
        output_dir = Path(config['output_dir'])
        os.makedirs(output_dir, exist_ok=True)
        
        output_file = output_dir / f"crew_{crew_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        output_data = {
            "crew_id": crew_id,
            "timestamp": datetime.now().isoformat(),
            "config": config,
            "results": results
        }
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        logger.info(f"Results saved to {output_file}")
    except Exception as e:
        logger.error(f"Error saving results: {str(e)}")

async def execute_crew(crew_id: str, config: Dict[str, Any]):
    """Execute a crew in the background."""
    try:
        # Initialize Ollama
        llm = Ollama(model=config['model'])
        
        # Create agents
        agents = [
            Agent(
                role=agent['role'],
                goal=agent['goal'],
                backstory=agent['backstory'],
                llm=llm,
                verbose=True
            )
            for agent in config['agents']
        ]
        
        # Create tasks
        tasks = [
            Task(
                description=task['description'],
                agent=agents[task['agent_index']]
            )
            for task in config['tasks']
        ]
        
        # Create and run crew
        crew = Crew(
            agents=agents,
            tasks=tasks,
            verbose=True
        )
        
        # Update crew status
        active_crews[crew_id]['status'] = 'running'
        
        # Execute crew
        results = await crew.execute()
        
        # Save results
        save_results(crew_id, results, config)
        
        # Update crew status
        active_crews[crew_id].update({
            'status': 'completed',
            'results': results,
            'completed_at': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error executing crew {crew_id}: {str(e)}")
        active_crews[crew_id].update({
            'status': 'failed',
            'error': str(e),
            'completed_at': datetime.now().isoformat()
        })

@app.post("/crew/create", response_model=CrewResponse)
async def create_crew(request: CrewRequest, background_tasks: BackgroundTasks):
    """Create and start a new crew."""
    try:
        # Generate crew ID
        crew_id = f"crew_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Prepare config
        config = {
            'topic': request.topic,
            'agents': [agent.dict() for agent in request.agents],
            'tasks': [task.dict() for task in request.tasks],
            'model': request.model,
            'output_dir': request.output_dir
        }
        
        # Store crew info
        active_crews[crew_id] = {
            'status': 'created',
            'config': config,
            'created_at': datetime.now().isoformat()
        }
        
        # Start crew execution in background
        background_tasks.add_task(execute_crew, crew_id, config)
        
        return CrewResponse(
            crew_id=crew_id,
            status='created',
            message="Crew created and execution started"
        )
        
    except Exception as e:
        logger.error(f"Error creating crew: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/crew/{crew_id}")
async def get_crew_status(crew_id: str):
    """Get the status of a crew."""
    if crew_id not in active_crews:
        raise HTTPException(status_code=404, detail="Crew not found")
    
    return active_crews[crew_id]

@app.get("/crew/list")
async def list_crews():
    """List all crews and their statuses."""
    return {
        crew_id: {
            'status': info['status'],
            'created_at': info['created_at'],
            'completed_at': info.get('completed_at')
        }
        for crew_id, info in active_crews.items()
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 