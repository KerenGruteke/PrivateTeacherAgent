from dotenv import load_dotenv

from src.utils.constants import CHAT_DEPLOYMENT_NAME, AZURE_OPENAI_ENDPOINT, API_VERSION
from src.utils.LLM_utils import LoggingAzureChatOpenAI

from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType

load_dotenv()
MAIN_PRIVATE_AGENT="MAIN_PRIVATE_AGENT"

# 1. Initialize LLM
llm = LoggingAzureChatOpenAI(
    agent_name=MAIN_PRIVATE_AGENT,
    azure_deployment=CHAT_DEPLOYMENT_NAME,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    openai_api_version=API_VERSION,
    openai_api_type="azure",
    temperature=0,
)



def root_calculator(number: int) -> str:
    return str(float(number)**0.5)

# 2. Define tools
search = DuckDuckGoSearchRun()
tools = [
    Tool(
        name="Search",
        func=search.run,
        description="Useful for answering questions about current or factual data"
    ),
    Tool(
        name="Calculator",
        func=root_calculator,
        description="Useful for math operations like square roots, multipication, etc."
    )
]

# 3. Create ReAct agent
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)

# 4. Run the agent
question = "What is the population of Paris, and what is the square root of that number?"
response = agent.run(question)

print("Answer:", response)

