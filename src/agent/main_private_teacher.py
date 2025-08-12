from dotenv import load_dotenv

from src.utils.constants import CHAT_DEPLOYMENT_NAME, AZURE_OPENAI_ENDPOINT, API_VERSION
from src.utils.LLM_utils import LoggingAzureChatOpenAI

from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
from src.agent.coacher import get_coacher_response
from src.agent.question_RAG import generate_question_agent
from src.agent.answer_evaluator import evaluate_answer

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


# 2. Define tools
tools = [
    Tool(
        name="Coacher",
        func=get_coacher_response,
        description=(
        "Produce and print a short, student-friendly motivational message based on the provided student_state. "
        "Use this to encourage the learner, explain the value of the current topic, or suggest a tiny next step. "
        "This tool does not return text — it prints directly for the student."
        ),
    ),
    Tool(
        name="Question RAG",
        func=generate_question_agent,
        description=(
            "Generate a practice question and its solution based on the user's request. "
            "First searches the internal subject-specific database for relevant material; "
            "if needed, can also search the web for additional context. "
            "Outputs a single JSON object with the question, correct solution, and optional hint/difficulty."
        )
    ),
    Tool(
        name="Answer Evaluator",
        func=evaluate_answer,
        description=(
            "Evaluate a student's answer against the reference solution. "
            "Returns JSON with correctness, score (0–1), feedback, common_mistakes. "
            "Also logs/merges common mistakes into the 'common_mistakes' DB for future use."
        ),
    ),
    
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

