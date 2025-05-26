from crewai import Agent, Task, Crew, Process
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List
from dotenv import load_dotenv

load_dotenv()

@CrewBase
class TestCrew:
    # agents: List[BaseAgent]
    # tasks: List[Task]

    # agents_config = 'config/agents.yaml'
    # tasks_config = 'config/tasks.yaml'
    
    @agent
    def research_agent(self) -> Agent:
        return Agent(
            config = self.agents_config['research_agent'],
            verbose=True
        )

    @agent
    def writer_agent(self) -> Agent:
        return Agent(
            config = self.agents_config['writer_agent'],
            verbose=True
        )

    @task
    def research_task(self) -> Task:
        return Task(
            config = self.tasks_config['research_task'],
            verbose=True
        )

    @task
    def writing_task(self) -> Task:
        return Task(
            config = self.tasks_config['writing_task'],
            verbose=True
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=[self.research_agent(), self.writer_agent()],
            tasks=[self.research_task(), self.writing_task()],
            # agents=self.agents,
            # tasks=self.tasks,
            verbose=True,
            process=Process.sequential,
        )

def main():
    test_topic = "Research on the Impact of Artificial Intelligence on Modern Business"
    
    try:
        test_crew = TestCrew()
        crew = test_crew.crew()
        result = crew.kickoff(inputs={"topic": test_topic})
        print("\nFinal Result:")
        print(result)
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main() 