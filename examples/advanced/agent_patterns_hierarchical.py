"""
Advanced example of implementing hierarchical agent patterns in CrewAI.
This demonstrates how to create complex agent hierarchies and task delegation.
"""

from crewai import Agent, Task, Crew
from langchain_ollama import OllamaLLM
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime
from pydantic import Field

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HierarchicalAgent(Agent):
    """A hierarchical agent that can delegate tasks to sub-agents."""
    
    sub_agents: List[Agent] = Field(default_factory=list)
    task_history: List[Dict[str, Any]] = Field(default_factory=list)
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def delegate_task(self, incoming_task: Task) -> str:
        """Delegate a task to appropriate sub-agents and aggregate results."""
        logger.info(f"Team Lead {self.role} delegating task: {incoming_task.description}")
        
        self.task_history.append({
            "task": incoming_task.description,
            "delegation_start_time": datetime.now(),
            "status": "delegation_in_progress"
        })
        
        try:
            relevant_agents = self._select_agents_for_task(incoming_task)
            
            aggregated_sub_task_outputs = []
            current_context_for_sub_agents = ""

            for sub_agent in relevant_agents:
                sub_task_description = (
                    f"As the {sub_agent.role}, your specific task is to contribute to the overall goal: "
                    f"'{incoming_task.description}'. Focus on your area of expertise. "
                    f"Previous context from team: {current_context_for_sub_agents if current_context_for_sub_agents else 'None'}"
                )
                
                sub_task_expected_output = (
                    f"A concise report from the {sub_agent.role} detailing your findings and work "
                    f"related to: {incoming_task.description}"
                )

                delegated_sub_task = Task(
                    description=sub_task_description,
                    agent=sub_agent,
                    expected_output=sub_task_expected_output
                )
                
                logger.info(f"Team Lead delegating to {sub_agent.role}: {delegated_sub_task.description}")
                temp_crew = Crew(
                    agents=[sub_agent],
                    tasks=[delegated_sub_task],
                    verbose=False
                )
                sub_agent_output = temp_crew.kickoff()
                aggregated_sub_task_outputs.append((sub_agent.role, sub_agent_output))
                current_context_for_sub_agents += f"\n\nOutput from {sub_agent.role}:\n{sub_agent_output}"

            final_result = self.aggregate_results_from_delegation(aggregated_sub_task_outputs)
            
            self.task_history[-1].update({
                "delegation_end_time": datetime.now(),
                "status": "delegation_completed",
                "result_summary": final_result[:200] + "..."
            })
            
            return final_result
        except Exception as e:
            logger.error(f"Error in task delegation by {self.role}: {str(e)}")
            self.task_history[-1].update({
                "delegation_end_time": datetime.now(),
                "status": "delegation_failed",
                "error": str(e)
            })
            raise
    
    def _select_agents_for_task(self, task: Task) -> List[Agent]:
        """Select appropriate sub-agents for a given task."""
        logger.info(f"Team Lead {self.role} selected sub-agents: {[sa.role for sa in self.sub_agents]} for task: {task.description}")
        return self.sub_agents
    
    def aggregate_results_from_delegation(self, sub_task_outputs: List[tuple[str, str]]) -> str:
        """Aggregate results from delegated sub-tasks."""
        logger.info(f"Team Lead {self.role} aggregating results from sub-agents.")
        formatted_results = "\n\n".join([
            f"Contribution from {role}:\n{result}"
            for role, result in sub_task_outputs
        ])
        return f"Aggregated Report coordinated by {self.role}:\n\n{formatted_results}"

class ResearchTeam:
    """A research team with hierarchical agent structure."""
    
    def __init__(self, topic: str):
        self.topic = topic
        self.setup_agents()
        self.setup_tasks()
    
    def setup_agents(self):
        """Set up the hierarchical agent structure."""
        llm = OllamaLLM(model="ollama/gemma:2b")

        self.researcher = Agent(
            role='Researcher',
            goal='Research topics thoroughly',
            backstory='Expert researcher with strong analytical skills',
            llm=llm
        )
        
        self.analyst = Agent(
            role='Analyst',
            goal='Analyze research findings',
            backstory='Data analyst with expertise in technology trends',
            llm=llm
        )
        
        self.writer = Agent(
            role='Writer',
            goal='Create engaging content',
            backstory='Experienced content writer',
            llm=llm
        )
        
        self.team_lead = HierarchicalAgent(
            role='Team Lead',
            goal='Coordinate research team efforts and delegate sub-tasks effectively',
            backstory='Experienced team leader, capable of breaking down complex tasks for specialized sub-agents.',
            llm=llm,
            sub_agents=[self.researcher, self.analyst, self.writer]
        )
    
    def setup_tasks(self):
        """Set up the research tasks for the Team Lead."""
        self.tasks = [
            Task(
                description=f"Coordinate the full research process for {self.topic}. Start by initiating detailed research, then analysis, then final report writing.",
                agent=self.team_lead,
                expected_output=f"A comprehensive final report on {self.topic}, incorporating contributions from all sub-agents, fully compiled and reviewed."
            )
        ]
    
    def execute_research(self) -> Dict[str, Any]:
        """Execute the research process using the Team Lead."""
        lead_crew = Crew(
            agents=[self.team_lead],
            tasks=self.tasks,
            verbose=True 
        )
        
        logger.info(f"ResearchTeam starting execution. Team Lead: {self.team_lead.role} will manage the process.")

        results = {}
        for task_for_team_lead in self.tasks:
            logger.info(f"ResearchTeam assigning high-level task to Team Lead: {task_for_team_lead.description}")
            try:
                final_task_result = self.team_lead.delegate_task(task_for_team_lead)
                results[task_for_team_lead.description] = final_task_result
            except Exception as e:
                logger.error(f"Error executing high-level task via Team Lead: {str(e)}")
                results[task_for_team_lead.description] = f"Error: {str(e)}"
        return results

def main():
    """Example usage of the hierarchical agent system."""
    logger.info("Starting main function for Hierarchical Agent Pattern example.")
    team = ResearchTeam(topic="AI in Education")
    
    logger.info("Executing research...")
    results = team.execute_research()
    
    print("\nResearch Results:")
    for task_desc, result_content in results.items():
        print(f"\nTask: {task_desc}")
        print(f"Result: {result_content}")
    
    print("\nTask History from Team Lead:")
    if hasattr(team.team_lead, 'task_history') and team.team_lead.task_history:
        for entry in team.team_lead.task_history:
            print(f"\n  Task Delegated: {entry.get('task')}")
            print(f"  Status: {entry.get('status')}")
            if entry.get('status') == 'completed' or entry.get('status') == 'delegation_completed':
                duration = entry.get('delegation_end_time', entry.get('end_time')) - entry.get('delegation_start_time', entry.get('start_time'))
                print(f"  Duration: {duration}")
            if 'error' in entry:
                print(f"  Error: {entry.get('error')}")
    else:
        print("  No task history recorded for Team Lead.")

if __name__ == "__main__":
    main() 