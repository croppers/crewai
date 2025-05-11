"""
Advanced example of implementing complex crew architecture in CrewAI.
This demonstrates sophisticated crew patterns with process management and monitoring.
"""

from crewai import Agent, Crew, Task, Process
from langchain_ollama import OllamaLLM
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime
import json
import os
from pathlib import Path
from pydantic import Field

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ResearchCrew(Crew):
    """A sophisticated research crew with advanced features."""
    
    topic: str = Field(...)
    output_dir: str = Field(default="output")
    
    researcher: Optional[Agent] = None
    analyst: Optional[Agent] = None
    writer: Optional[Agent] = None
    editor: Optional[Agent] = None
    results_file: Optional[Path] = None
    metrics: Optional[Dict[str, Any]] = None
    
    def __init__(self, **kwargs):
        # Extract topic from kwargs to pass it explicitly to agent/task creation methods.
        # 'topic' is a required field (Field(...)), so it should be in kwargs.
        current_topic = kwargs["topic"] 
        # output_dir will be handled by Pydantic when super().__init__ is called if it's also a field there,
        # or set on self by Pydantic if it's only a field of ResearchCrew.
        # For methods called after super().__init__ (like setup_environment), self.output_dir can be used.

        agents_list, agent_instances_map = self._create_agents_and_get_instances(topic=current_topic)
        tasks_list = self._create_tasks(topic=current_topic, agent_instances=agent_instances_map)
        
        super().__init__(agents=agents_list, tasks=tasks_list, **kwargs)
        
        self.researcher = agent_instances_map['researcher']
        self.analyst = agent_instances_map['analyst']
        self.writer = agent_instances_map['writer']
        self.editor = agent_instances_map['editor']
        
        self.setup_environment()
        self.setup_monitoring()
    
    def setup_environment(self):
        """Set up the working environment."""
        os.makedirs(self.output_dir, exist_ok=True)
        self.results_file = Path(self.output_dir) / f"research_{self.topic.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    def _create_agents_and_get_instances(self, topic: str) -> tuple[List[Agent], Dict[str, Agent]]:
        """Set up the specialized agents and return them as a list and map."""
        llm = OllamaLLM(model="ollama/gemma:2b")

        researcher_agent = Agent(
            role='Researcher',
            goal=f'Research {topic} thoroughly and gather comprehensive information',
            backstory='Expert researcher with a strong background in technology and AI',
            llm=llm
        )
        
        analyst_agent = Agent(
            role='Analyst',
            goal=f'Analyze research findings for {topic} and identify key insights',
            backstory='Data analyst with expertise in technology trends and market analysis',
            llm=llm
        )
        
        writer_agent = Agent(
            role='Writer',
            goal=f'Create engaging and informative content based on research and analysis of {topic}',
            backstory='Experienced content writer specializing in technology and AI topics',
            llm=llm
        )
        
        editor_agent = Agent(
            role='Editor',
            goal=f'Review and improve content about {topic} for clarity, accuracy, and engagement',
            backstory='Senior editor with experience in technology publications',
            llm=llm
        )
        
        agents_map = {
            "researcher": researcher_agent,
            "analyst": analyst_agent,
            "writer": writer_agent,
            "editor": editor_agent
        }
        return list(agents_map.values()), agents_map
    
    def _create_tasks(self, topic: str, agent_instances: Dict[str, Agent]) -> List[Task]:
        """Set up the research tasks with dependencies."""
        research_task = Task(
            description=f"Research {topic} thoroughly, focusing on recent developments and future implications.",
            agent=agent_instances['researcher'],
            expected_output=f"A comprehensive research report on {topic}, detailing recent developments, future implications, and gathered data."
        )
        analysis_task = Task(
            description=f"Analyze the research findings for {topic}, identify key trends, and evaluate potential impact. Use the report from the Researcher.",
            agent=agent_instances['analyst'],
            context=[research_task],
            expected_output=f"A detailed analysis of the research on {topic}, highlighting key trends, potential impact, and actionable insights."
        )
        writing_task = Task(
            description=f"Write a comprehensive report on {topic} incorporating research findings and analysis. Base this on the Researcher's report and the Analyst's insights.",
            agent=agent_instances['writer'],
            context=[research_task, analysis_task],
            expected_output=f"A well-written, engaging, and informative report on {topic}, clearly presenting research findings, analysis, and conclusions."
        )
        editing_task = Task(
            description=f"Review and edit the report on {topic} for clarity, accuracy, grammar, and engagement. Ensure it meets publication standards.",
            agent=agent_instances['editor'],
            context=[writing_task],
            expected_output=f"A polished, error-free, and publication-ready final version of the report on {topic}."
        )
        return [research_task, analysis_task, writing_task, editing_task]
    
    def setup_monitoring(self):
        """Set up monitoring and metrics collection."""
        self.metrics = {
            "topic": self.topic,
            "start_time": datetime.now(),
            "task_completion": {},
            "agent_performance": {},
            "errors": []
        }
    
    def execute_task(self, task: Task) -> str:
        """Execute a task with monitoring and error handling."""
        agent = task.agent
        if agent is None:
            error_msg = f"Task '{task.description}' has no agent assigned."
            logger.error(error_msg)
            self.metrics["errors"].append({
                "task_description": task.description,
                "error": error_msg,
                "timestamp": datetime.now()
            })
            return f"Error: {error_msg}"

        task_id = f"{agent.role.replace(' ', '_')}_{task.description[:30].replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        logger.info(f"Starting task: {task.description} by Agent: {agent.role}")
        self.metrics["task_completion"][task_id] = {
            "agent": agent.role,
            "description": task.description,
            "start_time": datetime.now(),
            "status": "in_progress"
        }
        
        try:
            result = agent.execute_task(task=task)
            
            self.metrics["task_completion"][task_id].update({
                "end_time": datetime.now(),
                "status": "completed",
            })
            
            if agent.role not in self.metrics["agent_performance"]:
                self.metrics["agent_performance"][agent.role] = {
                    "tasks_completed": 0,
                    "total_time_seconds": 0.0
                }
            
            self.metrics["agent_performance"][agent.role]["tasks_completed"] += 1
            duration = (self.metrics["task_completion"][task_id]["end_time"] - 
                       self.metrics["task_completion"][task_id]["start_time"])
            self.metrics["agent_performance"][agent.role]["total_time_seconds"] += duration.total_seconds()
            
            return result
        except Exception as e:
            logger.error(f"Error in task execution by {agent.role} for task '{task.description}': {str(e)}")
            self.metrics["task_completion"][task_id].update({
                "end_time": datetime.now(),
                "status": "failed",
                "error": str(e)
            })
            self.metrics["errors"].append({
                "task_id": task_id,
                "task_description": task.description,
                "agent": agent.role,
                "error": str(e),
                "timestamp": datetime.now()
            })
            raise
    
    def execute(self) -> Dict[str, Any]:
        """Execute the research process with monitoring.
        This custom execute method overrides the default Crew.kickoff() 
        to provide more detailed monitoring and result/metrics saving.
        """
        final_results_map = {}
        
        try:
            current_context_summary = ""
            for task_obj in self.tasks:
                logger.info(f"Processing task: {task_obj.description} for agent {task_obj.agent.role if task_obj.agent else 'None'}")
                task_result = self.execute_task(task_obj)
                final_results_map[task_obj.agent.role if task_obj.agent else f"task_{task_obj.description[:20]}"] = task_result
                current_context_summary += f"Result from {task_obj.agent.role if task_obj.agent else 'task'}: {task_result[:100]}...\n"

            self.save_results(final_results_map)
            return final_results_map
        except Exception as e:
            logger.error(f"Error in custom crew execution: {str(e)}")
            self.metrics["errors"].append({
                "type": "crew_execution_overall",
                "error": str(e),
                "timestamp": datetime.now()
            })
            self.save_results(final_results_map if final_results_map else {"error": "Execution failed"})
            raise
        finally:
            if self.metrics:
                self.metrics["end_time"] = datetime.now()
                self.save_metrics()
    
    def save_results(self, results: Dict[str, Any]):
        """Save research results to file."""
        if not self.results_file:
            logger.warning("Results file path not set, cannot save results.")
            return
        try:
            with open(self.results_file, 'w') as f:
                serializable_results = {}
                for k, v in results.items():
                    if isinstance(v, datetime):
                        serializable_results[k] = v.isoformat()
                    else:
                        serializable_results[k] = v

                json.dump({
                    "topic": self.topic,
                    "timestamp": datetime.now().isoformat(),
                    "results": serializable_results
                }, f, indent=2)
            logger.info(f"Results saved to {self.results_file}")
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")
    
    def save_metrics(self):
        """Save execution metrics to file."""
        if not self.metrics or not self.output_dir:
            logger.warning("Metrics or output directory not initialized, cannot save metrics.")
            return
            
        metrics_file_name = f"metrics_{self.topic.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        metrics_file_path = Path(self.output_dir) / metrics_file_name

        try:
            serializable_metrics = json.loads(json.dumps(self.metrics, default=str))

            if 'start_time' in serializable_metrics and isinstance(serializable_metrics['start_time'], datetime):
                 serializable_metrics['start_time'] = serializable_metrics['start_time'].isoformat()
            if 'end_time' in serializable_metrics and isinstance(serializable_metrics['end_time'], datetime):
                 serializable_metrics['end_time'] = serializable_metrics['end_time'].isoformat()

            if "task_completion" in serializable_metrics:
                for task_id, details in serializable_metrics["task_completion"].items():
                    if 'start_time' in details and isinstance(details['start_time'], datetime):
                        details['start_time'] = details['start_time'].isoformat()
                    if 'end_time' in details and isinstance(details['end_time'], datetime):
                        details['end_time'] = details['end_time'].isoformat()
            
            if "errors" in serializable_metrics:
                for error_entry in serializable_metrics["errors"]:
                    if 'timestamp' in error_entry and isinstance(error_entry['timestamp'], datetime):
                        error_entry['timestamp'] = error_entry['timestamp'].isoformat()

            with open(metrics_file_path, 'w') as f:
                json.dump(serializable_metrics, f, indent=2)
            logger.info(f"Metrics saved to {metrics_file_path}")
        except Exception as e:
            logger.error(f"Error saving metrics: {str(e)}")

def main():
    """Example usage of the complex research crew."""
    crew = ResearchCrew(
        topic="AI in Healthcare",
        output_dir="research_output_complex",
        process=Process.sequential,
        verbose=True
    )
    
    try:
        results = crew.execute()
        
        print("\nResearch Results:")
        if results:
            for role, result in results.items():
                print(f"\nRole/Task: {role}")
                print(f"Output: {result}")
        else:
            print("No results produced.")
        
        print("\nExecution Metrics:")
        if crew.metrics and "agent_performance" in crew.metrics:
            for agent_name, agent_metrics in crew.metrics["agent_performance"].items():
                print(f"\nAgent: {agent_name}:")
                print(f"  Tasks completed: {agent_metrics.get('tasks_completed', 0)}")
                print(f"  Total time: {agent_metrics.get('total_time_seconds', 0):.2f} seconds")
        
        if crew.metrics and crew.metrics.get("errors"):
            print("\nErrors encountered during execution:")
            for error_detail in crew.metrics["errors"]:
                print(f"- Agent: {error_detail.get('agent', 'N/A')}, Task: {error_detail.get('task_description', 'N/A')}, Error: {error_detail.get('error', 'Unknown')}")
    except Exception as e:
        print(f"\nCritical Error in research execution: {str(e)}")
        if hasattr(crew, 'metrics') and crew.metrics and hasattr(crew, 'save_metrics'):
            print("Attempting to save metrics on critical failure...")
            crew.save_metrics()

if __name__ == "__main__":
    main() 