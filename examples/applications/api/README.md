# CrewAI API Application

This example demonstrates how to create a FastAPI application that provides an API for running CrewAI crews with Ollama. The API allows you to create and manage crews, execute tasks, and monitor their progress.

## Features

- Create and manage multiple crews
- Execute crews asynchronously
- Monitor crew status and results
- Save crew execution results to files
- RESTful API endpoints for crew management

## Setup

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Make sure Ollama is running with the desired model (e.g., gemma:2b):
```bash
ollama run gemma:2b
```

3. Start the API server:
```bash
python main.py
```

The server will start on `http://localhost:8000`.

## API Endpoints

### Create a Crew
```http
POST /crew/create
```

Request body:
```json
{
    "topic": "AI in Healthcare",
    "agents": [
        {
            "role": "Researcher",
            "goal": "Research AI applications in healthcare",
            "backstory": "Expert in healthcare technology research"
        },
        {
            "role": "Writer",
            "goal": "Write a comprehensive report",
            "backstory": "Experienced technical writer"
        }
    ],
    "tasks": [
        {
            "description": "Research current AI applications in healthcare",
            "agent_index": 0
        },
        {
            "description": "Write a report on the findings",
            "agent_index": 1
        }
    ],
    "model": "gemma:2b",
    "output_dir": "api_output"
}
```

### Get Crew Status
```http
GET /crew/{crew_id}
```

### List All Crews
```http
GET /crew/list
```

## Example Usage

Using curl:
```bash
# Create a new crew
curl -X POST http://localhost:8000/crew/create \
    -H "Content-Type: application/json" \
    -d '{
        "topic": "AI in Healthcare",
        "agents": [
            {
                "role": "Researcher",
                "goal": "Research AI applications in healthcare",
                "backstory": "Expert in healthcare technology research"
            },
            {
                "role": "Writer",
                "goal": "Write a comprehensive report",
                "backstory": "Experienced technical writer"
            }
        ],
        "tasks": [
            {
                "description": "Research current AI applications in healthcare",
                "agent_index": 0
            },
            {
                "description": "Write a report on the findings",
                "agent_index": 1
            }
        ]
    }'

# Get crew status
curl http://localhost:8000/crew/crew_20240101_120000

# List all crews
curl http://localhost:8000/crew/list
```

## Output

The API saves crew execution results in the specified output directory (default: `api_output`). Each result file contains:
- Crew ID
- Timestamp
- Configuration
- Execution results

## Error Handling

The API includes comprehensive error handling:
- Input validation using Pydantic models
- HTTP exceptions for common errors
- Detailed error logging
- Status tracking for crew execution

## Notes

- Crews are executed asynchronously in the background
- Results are saved to files for persistence
- The API maintains an in-memory state of active crews
- All operations are logged for monitoring and debugging 