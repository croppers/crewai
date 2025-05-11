# Building Agentic AI with Low-Cost LLMs: A Comprehensive Tutorial

## Overview
This tutorial guides you through building agentic AI applications using CrewAI and Ollama, focusing on cost-effective implementation with local LLMs. The tutorial is split into two 4-hour sessions, providing both theoretical knowledge and hands-on experience.

## Prerequisites
- Python 3.9+
- Git
- Basic understanding of Python programming
- Basic understanding of AI/ML concepts

## Quick Start
1. Clone this repository:
```bash
git clone [repository-url]
cd crewai-tutorial
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install Ollama:
- Visit [ollama.com](https://ollama.com)
- Download and install for your operating system
- Pull a base model:
```bash
ollama pull gemma:2b
```

## Tutorial Structure

### Part 1: Foundations & Basic Implementation
- [Session 1 Guide](docs/part1/session1.md)
  - Environment Setup
  - CrewAI Fundamentals
  - Ollama Deep Dive
  - Building Your First Agent
  - Multi-Agent Basics

### Part 2: Advanced Implementation & Real-World Applications
- [Session 2 Guide](docs/part2/session2.md)
  - Advanced Agent Development
  - Building Complex Crews
  - Real-World Application Development
  - Production & Beyond

## Code Examples
All code examples are organized in the `examples` directory:
- [Basic Examples](examples/basic/)
- [Advanced Examples](examples/advanced/)
- [Complete Applications](examples/applications/)

## Resources
- [CrewAI Documentation](https://docs.crewai.com)
- [Ollama Documentation](https://ollama.com/docs)
- [Model Comparison](docs/resources/model-comparison.md)
- [Cost Analysis](docs/resources/cost-analysis.md)

## Contributing
Feel free to submit issues and enhancement requests!

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. 