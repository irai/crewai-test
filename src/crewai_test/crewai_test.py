from crewai import Agent, Task, Crew, Process
from crewai.project import CrewBase, agent, crew, task
from typing import List
from dotenv import load_dotenv

load_dotenv()

@CrewBase
class TestCrew:
    # agents: List[BaseAgent]
    # tasks: List[Task]

    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'
    
    @agent
    def research_agent(self) -> Agent:
        return Agent(
            config = self.agents_config['research_agent'],
            verbose=True,
            allow_delegation=True
        )

    @agent
    def writer_agent(self) -> Agent:
        return Agent(
            config = self.agents_config['writer_agent'],
            verbose=True,
            allow_delegation=True
        )

    @task
    def research_task(self) -> Task:
        return Task(
            config = self.tasks_config['research_task'],
        )

    @task
    def writing_task(self) -> Task:
        return Task(
            config = self.tasks_config['writing_task'],
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=[self.research_agent(), self.writer_agent()],
            tasks=[self.research_task(), self.writing_task()],
            verbose=True,
            # process=Process.sequential,
            process=Process.hierarchical,
            manager_llm="openai/gpt-4o"
        )

def main():
    test_topic = "use a research agent to research the Impact of AI Agents on Enterprise Architecture. return the raw research data"
    
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