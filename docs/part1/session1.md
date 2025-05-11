# Part 1: Foundations & Basic Implementation

## Session 1: Environment Setup & Basic Concepts

### 1. Welcome & Overview (15 minutes)

#### Introduction to CrewAI and Ollama
- **CrewAI**: A framework for building agentic AI applications
  - Agent-based architecture
  - Collaborative AI systems
  - Tool integration capabilities
  - Process management

- **Ollama**: Local LLM deployment
  - Open-source model hosting
  - Cost-effective inference
  - Model management
  - Performance optimization

#### Why Local LLMs Matter
- Cost reduction
- Data privacy
- Latency improvement
- Customization potential
- Offline capabilities

#### Cost Comparison Overview
- Cloud LLM costs (GPT-4, Claude, etc.)
- Local LLM costs (Ollama)
- Infrastructure requirements
- Total cost of ownership

### 2. Environment Setup (45 minutes)

#### Installing Ollama
1. **Download and Installation**
   ```bash
   # macOS
   curl -fsSL https://ollama.com/install.sh | sh
   
   # Linux
   curl -fsSL https://ollama.com/install.sh | sh
   
   # Windows
   # Download from https://ollama.com/download
   ```

2. **Basic Model Testing**
   ```bash
   # Pull a base model
   ollama pull gemma:2b
   
   # Test the model
   ollama run gemma:2b "Hello, how are you?"
   ```

3. **Model Management**
   ```bash
   # List available models
   ollama list
   
   # Remove a model
   ollama rm gemma:2b
   
   # Pull specific model version
   ollama pull gemma:2b
   ```

#### Setting up CrewAI
1. **Python Environment Setup**
   ```bash
   # Create virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # Install dependencies
   pip install -r requirements.txt
   ```

2. **Basic Configuration**
   Create `.env` file:
   ```env
   OLLAMA_BASE_URL=http://localhost:11434
   DEFAULT_MODEL=ollama/gemma:2b
   ```

3. **Project Structure**
   ```
   tutorial/
   ├── src/
   │   ├── agents/
   │   │   ├── __init__.py
   │   │   └── base_agent.py
   │   ├── tools/
   │   │   ├── __init__.py
   │   │   └── base_tools.py
   │   └── config/
   │       └── settings.py
   ├── examples/
   │   └── basic/
   │       └── hello_world.py
   └── tests/
       └── test_basic.py
   ```

### 3. CrewAI Fundamentals (30 minutes)

#### Agent Architecture
```python
# examples/basic/hello_world.py
from crewai import Agent
from langchain_community.llms import Ollama

# Initialize Ollama
llm = Ollama(model="ollama/gemma:2b")

# Create a basic agent
agent = Agent(
    role='Researcher',
    goal='Research and analyze topics effectively',
    backstory='Expert researcher with years of experience',
    llm=llm,
    verbose=True
)

# Example task
task = agent.execute("Research the impact of AI on healthcare")
```

#### Crew Concepts
```python
# examples/basic/first_crew.py
from crewai import Agent, Crew, Task
from langchain_community.llms import Ollama

# Initialize agents
researcher = Agent(
    role='Researcher',
    goal='Research topics thoroughly',
    backstory='Expert researcher',
    llm=Ollama(model="ollama/gemma:2b")
)

writer = Agent(
    role='Writer',
    goal='Create engaging content',
    backstory='Experienced content writer',
    llm=Ollama(model="ollama/gemma:2b")
)

# Create tasks
research_task = Task(
    description="Research AI in healthcare",
    agent=researcher
)

writing_task = Task(
    description="Write a blog post about AI in healthcare",
    agent=writer
)

# Create and run crew
crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, writing_task]
)

result = crew.kickoff()
```

### 4. Ollama Deep Dive (30 minutes)

#### Available Models
- Gemma (2B, 7B)
- Llama2 (7B, 13B, 70B)
- CodeLlama
- Neural Chat
- Vicuna

#### Model Selection Criteria
- Task requirements
- Hardware constraints
- Performance needs
- Cost considerations

#### Performance Considerations
- Memory usage
- Inference speed
- Quality of outputs
- Resource utilization

#### Cost Implications
- Hardware requirements
- Electricity costs
- Maintenance overhead
- Scaling considerations

### 5. Building Your First Agent (45 minutes)

#### Single Agent Implementation
```python
# examples/basic/custom_agent.py
from crewai import Agent, Task
from langchain_community.llms import Ollama
from langchain.tools import Tool

# Custom tool example
def search_web(query: str) -> str:
    # Implement web search functionality
    return f"Search results for: {query}"

# Create custom tools
tools = [
    Tool(
        name="WebSearch",
        func=search_web,
        description="Search the web for information"
    )
]

# Create agent with custom tools
agent = Agent(
    role='Research Analyst',
    goal='Analyze and research topics thoroughly',
    backstory='Expert analyst with strong research skills',
    llm=Ollama(model="ollama/gemma:2b"),
    tools=tools,
    verbose=True
)

# Execute task
task = Task(
    description="Research the latest developments in quantum computing",
    agent=agent
)

result = agent.execute(task)
```

### 6. Multi-Agent Basics (30 minutes)

#### Introduction to Crews
- Crew architecture
- Agent communication
- Task delegation
- Collaboration patterns

#### Building a Simple Crew
```python
# examples/basic/research_crew.py
from crewai import Agent, Crew, Task
from langchain_community.llms import Ollama

# Initialize agents
researcher = Agent(
    role='Researcher',
    goal='Research topics thoroughly',
    backstory='Expert researcher',
    llm=Ollama(model="ollama/gemma:2b")
)

analyst = Agent(
    role='Analyst',
    goal='Analyze research findings',
    backstory='Data analyst with strong analytical skills',
    llm=Ollama(model="ollama/gemma:2b")
)

writer = Agent(
    role='Writer',
    goal='Create engaging content',
    backstory='Experienced content writer',
    llm=Ollama(model="ollama/gemma:2b")
)

# Create tasks
research_task = Task(
    description="Research AI in healthcare",
    agent=researcher
)

analysis_task = Task(
    description="Analyze the research findings",
    agent=analyst
)

writing_task = Task(
    description="Write a comprehensive report",
    agent=writer
)

# Create and run crew
crew = Crew(
    agents=[researcher, analyst, writer],
    tasks=[research_task, analysis_task, writing_task],
    verbose=True
)

result = crew.kickoff()
```

## Hands-on Exercises

### Exercise 1: Basic Agent Setup
1. Create a virtual environment
2. Install required packages
3. Set up Ollama with Gemma 2B model
4. Create a simple agent
5. Execute a basic task

### Exercise 2: Custom Tools
1. Implement a custom web search tool
2. Create an agent with the custom tool
3. Test the agent with different queries

### Exercise 3: Multi-Agent System
1. Create a research crew
2. Implement task delegation
3. Test the crew with a complex task

## Resources
- [CrewAI Documentation](https://docs.crewai.com)
- [Ollama Documentation](https://ollama.com/docs)
- [Gemma Model Card](https://huggingface.co/google/gemma-2b)
- [Model Comparison](docs/resources/model-comparison.md)
- [Cost Analysis](docs/resources/cost-analysis.md)

## Next Steps
- Review the code examples in the `examples/basic/` directory
- Complete the hands-on exercises
- Prepare for Session 2: Advanced Implementation 