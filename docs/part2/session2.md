# Part 2: Advanced Implementation & Real-World Applications

## Session 2: Advanced Concepts & Production Deployment

### 1. Advanced Agent Development (60 minutes)

#### Custom Tools Development
```python
# examples/advanced/custom_tools.py
from typing import List, Dict, Any
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
import requests
from bs4 import BeautifulSoup

class WebScraperTool(BaseTool):
    name = "web_scraper"
    description = "Scrape content from a website"
    
    class InputSchema(BaseModel):
        url: str = Field(..., description="URL to scrape")
        selector: str = Field(..., description="CSS selector for content")
    
    def _run(self, url: str, selector: str) -> str:
        try:
            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            content = soup.select(selector)
            return "\n".join([elem.text for elem in content])
        except Exception as e:
            return f"Error scraping website: {str(e)}"

class DataAnalysisTool(BaseTool):
    name = "data_analyzer"
    description = "Analyze data and generate insights"
    
    class InputSchema(BaseModel):
        data: List[Dict[str, Any]] = Field(..., description="Data to analyze")
        metrics: List[str] = Field(..., description="Metrics to calculate")
    
    def _run(self, data: List[Dict[str, Any]], metrics: List[str]) -> str:
        # Implement data analysis logic
        return "Analysis results..."

# Create advanced agent with custom tools
from crewai import Agent
from langchain_community.llms import Ollama

advanced_agent = Agent(
    role='Data Analyst',
    goal='Analyze data and generate insights',
    backstory='Expert data analyst with strong analytical skills',
    llm=Ollama(model="mistral"),
    tools=[WebScraperTool(), DataAnalysisTool()],
    verbose=True
)
```

#### Advanced Agent Patterns
```python
# examples/advanced/agent_patterns.py
from crewai import Agent, Task, Crew
from langchain_community.llms import Ollama
from typing import List, Dict
import asyncio

class HierarchicalAgent(Agent):
    def __init__(self, sub_agents: List[Agent], **kwargs):
        super().__init__(**kwargs)
        self.sub_agents = sub_agents
    
    async def delegate_task(self, task: Task) -> str:
        # Implement task delegation logic
        results = []
        for agent in self.sub_agents:
            result = await agent.execute(task)
            results.append(result)
        return self.aggregate_results(results)
    
    def aggregate_results(self, results: List[str]) -> str:
        # Implement result aggregation logic
        return "\n".join(results)

# Create hierarchical agent system
researcher = Agent(
    role='Researcher',
    goal='Research topics thoroughly',
    llm=Ollama(model="mistral")
)

analyst = Agent(
    role='Analyst',
    goal='Analyze research findings',
    llm=Ollama(model="mistral")
)

writer = Agent(
    role='Writer',
    goal='Create engaging content',
    llm=Ollama(model="mistral")
)

team_lead = HierarchicalAgent(
    role='Team Lead',
    goal='Coordinate team efforts',
    backstory='Experienced team leader',
    llm=Ollama(model="mistral"),
    sub_agents=[researcher, analyst, writer]
)
```

### 2. Building Complex Crews (60 minutes)

#### Advanced Crew Architecture
```python
# examples/advanced/complex_crew.py
from crewai import Agent, Crew, Task, Process
from langchain_community.llms import Ollama
from typing import List, Dict
import asyncio

class ResearchCrew(Crew):
    def __init__(self, topic: str, **kwargs):
        super().__init__(**kwargs)
        self.topic = topic
        self.setup_agents()
        self.setup_tasks()
    
    def setup_agents(self):
        self.researcher = Agent(
            role='Researcher',
            goal='Research topics thoroughly',
            llm=Ollama(model="mistral")
        )
        
        self.analyst = Agent(
            role='Analyst',
            goal='Analyze research findings',
            llm=Ollama(model="mistral")
        )
        
        self.writer = Agent(
            role='Writer',
            goal='Create engaging content',
            llm=Ollama(model="mistral")
        )
        
        self.editor = Agent(
            role='Editor',
            goal='Review and improve content',
            llm=Ollama(model="mistral")
        )
    
    def setup_tasks(self):
        self.tasks = [
            Task(
                description=f"Research {self.topic}",
                agent=self.researcher
            ),
            Task(
                description="Analyze research findings",
                agent=self.analyst
            ),
            Task(
                description="Write comprehensive report",
                agent=self.writer
            ),
            Task(
                description="Review and edit content",
                agent=self.editor
            )
        ]
    
    async def execute(self) -> Dict[str, str]:
        results = {}
        for task in self.tasks:
            result = await task.execute()
            results[task.agent.role] = result
        return results

# Create and run complex crew
crew = ResearchCrew(
    topic="AI in Healthcare",
    process=Process.sequential,
    verbose=True
)

results = asyncio.run(crew.execute())
```

#### Optimization & Performance
```python
# examples/advanced/optimization.py
from crewai import Agent, Crew, Task
from langchain_community.llms import Ollama
from typing import List, Dict
import time
import logging

class OptimizedCrew(Crew):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.setup_logging()
        self.setup_monitoring()
    
    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def setup_monitoring(self):
        self.metrics = {
            'token_usage': 0,
            'execution_time': 0,
            'task_completion': {}
        }
    
    async def execute_task(self, task: Task) -> str:
        start_time = time.time()
        try:
            result = await task.execute()
            self.metrics['task_completion'][task.agent.role] = True
            self.metrics['execution_time'] += time.time() - start_time
            return result
        except Exception as e:
            self.logger.error(f"Task failed: {str(e)}")
            self.metrics['task_completion'][task.agent.role] = False
            raise
```

### 3. Real-World Application Development (60 minutes)

#### Project Planning
```python
# examples/applications/content_team/main.py
from crewai import Agent, Crew, Task, Process
from langchain_community.llms import Ollama
from typing import List, Dict
import asyncio
import json
import os

class ContentTeam:
    def __init__(self, config_path: str):
        self.config = self.load_config(config_path)
        self.setup_environment()
        self.setup_crew()
    
    def load_config(self, config_path: str) -> Dict:
        with open(config_path, 'r') as f:
            return json.load(f)
    
    def setup_environment(self):
        os.makedirs(self.config['output_dir'], exist_ok=True)
        self.setup_logging()
    
    def setup_crew(self):
        self.crew = Crew(
            agents=self.create_agents(),
            tasks=self.create_tasks(),
            process=Process.sequential,
            verbose=True
        )
    
    def create_agents(self) -> List[Agent]:
        return [
            Agent(
                role=role,
                goal=details['goal'],
                backstory=details['backstory'],
                llm=Ollama(model=self.config['model'])
            )
            for role, details in self.config['agents'].items()
        ]
    
    def create_tasks(self) -> List[Task]:
        return [
            Task(
                description=task['description'],
                agent=self.crew.agents[task['agent']]
            )
            for task in self.config['tasks']
        ]
    
    async def run(self) -> Dict[str, str]:
        return await self.crew.execute()

# Configuration file (config.json)
"""
{
    "model": "mistral",
    "output_dir": "output",
    "agents": {
        "researcher": {
            "goal": "Research topics thoroughly",
            "backstory": "Expert researcher"
        },
        "writer": {
            "goal": "Create engaging content",
            "backstory": "Experienced writer"
        }
    },
    "tasks": [
        {
            "description": "Research AI in healthcare",
            "agent": "researcher"
        },
        {
            "description": "Write blog post",
            "agent": "writer"
        }
    ]
}
"""
```

### 4. Production & Beyond (60 minutes)

#### Deployment & Scaling
```python
# examples/applications/api/main.py
from fastapi import FastAPI, HTTPException
from crewai import Agent, Crew, Task
from langchain_community.llms import Ollama
from pydantic import BaseModel
from typing import List, Dict
import uvicorn
import asyncio
import logging

app = FastAPI(title="CrewAI API")

class CrewRequest(BaseModel):
    topic: str
    agents: List[Dict[str, str]]
    tasks: List[Dict[str, str]]

@app.post("/crew/execute")
async def execute_crew(request: CrewRequest):
    try:
        crew = Crew(
            agents=[
                Agent(
                    role=agent['role'],
                    goal=agent['goal'],
                    llm=Ollama(model="mistral")
                )
                for agent in request.agents
            ],
            tasks=[
                Task(
                    description=task['description'],
                    agent=crew.agents[task['agent_index']]
                )
                for task in request.tasks
            ],
            verbose=True
        )
        
        result = await crew.execute()
        return {"status": "success", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

#### Monitoring & Observability
```python
# examples/applications/monitoring/monitor.py
from prometheus_client import start_http_server, Counter, Histogram
import time
import logging
from typing import Dict, Any

class CrewMonitor:
    def __init__(self):
        self.setup_metrics()
        self.setup_logging()
    
    def setup_metrics(self):
        self.task_counter = Counter(
            'crew_task_total',
            'Total number of tasks executed',
            ['agent', 'status']
        )
        
        self.execution_time = Histogram(
            'crew_task_duration_seconds',
            'Time spent executing tasks',
            ['agent']
        )
        
        self.token_usage = Counter(
            'crew_token_usage_total',
            'Total tokens used',
            ['agent']
        )
    
    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def record_task(self, agent: str, status: str):
        self.task_counter.labels(agent=agent, status=status).inc()
    
    def record_execution_time(self, agent: str, duration: float):
        self.execution_time.labels(agent=agent).observe(duration)
    
    def record_token_usage(self, agent: str, tokens: int):
        self.token_usage.labels(agent=agent).inc(tokens)

# Start monitoring server
if __name__ == "__main__":
    monitor = CrewMonitor()
    start_http_server(8000)
```

## Hands-on Exercises

### Exercise 1: Custom Tools Development
1. Implement a custom web scraping tool
2. Create a data analysis tool
3. Integrate tools with an agent
4. Test the agent with different tasks

### Exercise 2: Complex Crew Implementation
1. Create a hierarchical agent system
2. Implement task delegation
3. Add monitoring and logging
4. Test the system with complex tasks

### Exercise 3: Production Deployment
1. Create a FastAPI application
2. Implement monitoring
3. Add error handling
4. Deploy the application

## Resources
- [CrewAI Documentation](https://docs.crewai.com)
- [Ollama Documentation](https://ollama.com/docs)
- [FastAPI Documentation](https://fastapi.tiangolo.com)
- [Prometheus Documentation](https://prometheus.io/docs)

## Next Steps
- Review the advanced examples in the `examples/advanced/` directory
- Complete the hands-on exercises
- Explore the production applications in `examples/applications/`
- Consider contributing to the project 