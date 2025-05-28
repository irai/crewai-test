from crewai import Agent, Task, Crew, Process
from typing import List
from dotenv import load_dotenv

load_dotenv()

class TestCrew:
    def research_agent(self) -> Agent:
        return Agent(
            role="Research Specialist",
            goal="Gather and analyze information effectively",
            backstory="""You are an expert researcher with a keen eye for detail.
            Your expertise lies in finding relevant information and organizing it
            in a clear, structured manner.""",
            llm="openai/gpt-4o",
            verbose=True,
            allow_delegation=True
        )

    def writer_agent(self) -> Agent:
        return Agent(
            role="Content Writer",
            goal="Create clear and engaging content based on research",
            backstory="""You are a skilled writer who excels at transforming
            research into compelling content. You have a talent for making
            complex information accessible and engaging.""",
            llm="openai/gpt-4o",
            verbose=True,
            allow_delegation=True
        )

    def research_task(self) -> Task:
        return Task(
            description="""Research the following topic and provide key findings:
            {topic}
            
            Your research should include:
            1. Main points and key facts
            2. Supporting evidence
            3. Any relevant statistics or data
            
            Format your findings in a clear, structured manner.""",
            expected_output="""A comprehensive research report containing key findings, supporting evidence,
            and relevant statistics about the topic, formatted in a clear and structured manner.""",
            agent=self.research_agent()
        )

    def writing_task(self) -> Task:
        return Task(
            description="""Using the research provided, create a comprehensive article about:
            {topic}
            
            The article should:
            1. Have a clear introduction
            2. Include all key points from the research
            3. Be well-structured and engaging
            4. End with a strong conclusion
            
            Research findings:""",
            expected_output="""A well-written, engaging article that effectively communicates the research findings
            with a clear structure, including introduction, key points, and conclusion.""",
            agent=self.writer_agent(),
            context=['research_task']
        )

    def create_crew(self) -> Crew:
        return Crew(
            agents=[self.research_agent(), self.writer_agent()],
            tasks=[self.research_task()],
            verbose=True,
            process=Process.sequential,
            manager_llm="openai/gpt-4o"
        )

def main():
    test_topic = "use a research agent to research the Impact of AI Agents on Enterprise Architecture. return the raw research data"
    
    try:
        test_crew = TestCrew()
        
        # result = test_crew.research_agent().kickoff(test_topic)
        # print("\nFinal Result:\n", result)
        # result = test_crew.writer_agent().kickoff(test_topic)
        # print("\nFinal Result:\n", result)
        # result = test_crew.research_task().execute({"topic": test_topic})
        # result = test_crew.research_agent().kickoff(test_topic)
        
        # Example of using the full crew
        crew = test_crew.create_crew()
        result = crew.kickoff(inputs={"topic": test_topic})
        print("\nFinal Result:\n", result)
        
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main() 