
from crewai import Agent, Crew, Task, Process
from dotenv import load_dotenv

load_dotenv()

llm = "openai/gpt-4o-mini"

# Create collaborative agents
agent1 = Agent(
    role="City Specialist",
    goal="Find accurate, up-to-date information on city and its surroundings",
    backstory="""You're a meticulous researcher with expertise in finding 
    reliable sources and fact-checking information across various domains.""",
    allow_delegation=False,
    verbose=True,
    llm=llm
)

# agent1_task = Task(
#     description="""Find accurate, up-to-date information on {city} and its surroundings""",
#     expected_output="A list of facts about the city",
#     agent=agent1
# )

agent2 = Agent(
    role="Food Specialist",
    goal="Find accurate, up-to-date information on regarding the traditional food in a city",
    backstory="""You're a meticulous researcher with expertise in finding 
    reliable sources and fact-checking information across various domains.""",
    allow_delegation=False,
    verbose=True,
    llm=llm
)

# agent2_task = Task(
#     description="""Find accurate, up-to-date information on regarding the traditional food in the {city}""",
#     expected_output="A list of facts about the traditional food in the city",
#     agent=agent2
# )

agent3 = Agent(
    role="Restaurant Reviewer",
    goal="Write a review the traditional food in a city",
    backstory="""You're an experienced reviewer with with experience in reviewing restaurants and food.
    You're known for your detailed and honest reviews.""",
    allow_delegation=True,
    verbose=True,
    llm="openai/gpt-4o"
)

# Create a task that encourages collaboration
article_task = Task(
    description="""Write a review about the traditional food in {city}.
    
    The article should include:
    - Facts about the city based on your research so you can provide some context about the city
    - Best traditional food in the city
    
    Collaborate with your teammates to ensure accuracy and quality.
    """,
    expected_output="A review about the traditional food in the city",
    agent=agent3,
    # context=[agent1, agent2]
)

# Create collaborative crew
crew = Crew(
    agents=[agent1, agent2, agent3],
    tasks=[article_task],
    process=Process.sequential,
    verbose=True
)

result = crew.kickoff({"city": "Singapore"})
print(result)
