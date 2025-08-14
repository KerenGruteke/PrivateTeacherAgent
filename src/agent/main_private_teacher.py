from dotenv import load_dotenv

from src.utils.constants import CHAT_DEPLOYMENT_NAME, AZURE_OPENAI_ENDPOINT, API_VERSION
from src.utils.LLM_utils import LoggingAzureChatOpenAI

from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
from src.agent.coacher import get_coacher_response
from src.agent.question_RAG import generate_question_agent
from src.agent.answer_evaluator import evaluate_answer
from src.agent.hand_in_hand_solver import hand_in_hand_agent
from src.agent.prompts import (
    WELCOME_PROMPT,
    INITIALIZE_MAIN_PRIVATE_TEACHER_SYSTEM_PROMPT,
    INITIALIZE_MAIN_PRIVATE_TEACHER_USER_PROMPT
)

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

def get_student_answer(question: str):
    """
    Tool Name: Get Student Answer
    Description:
        Prompts the student to provide an answer to a given question.
        This tool is intended to be used interactively during student interaction
        where the AI tutor persent the question to the student collects answers from the student.

    Args:
        question (str): The question to ask the student.

    Returns:
        str: The student's answer as plain text.
    """
    return str(input(f"\nQuestion: {question}\nWrite your answer here: "))

# 2. Define tools
tools = [
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
        name="Get Student Answer",
        func=get_student_answer,
        description=("Prompts the student to provide an answer to the current sub-question. "
                     "Returns the student's answer as a string."
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
    Tool(
        name="Hand In Hand",
        func=hand_in_hand_agent,
        description=(
            "An interactive tutoring tool that guides a student through a question step-by-step. "
            "It breaks the question into smaller sub-steps, evaluates the student's answers, "
            "and provides constructive, personalized feedback to help the student improve."
        ),
    ),
    Tool(
        name="Coacher",
        func=get_coacher_response,
        description=(
        "Produce and print a short, student-friendly motivational message based on the provided student_state. "
        "Use this to encourage the learner, explain the value of the current course, or suggest a tiny next step. "
        "This tool does not return text — it prints directly for the student."
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

prompt = INITIALIZE_HAND_IN_HAND_SYSTEM_PROMPT + "\n" + \
         INITIALIZE_HAND_IN_HAND_USER_PROMPT.format(
             course=course,
             question=question,
             reference_solution=solution,
             student_answer=student_answer
         )

print(WELCOME_PROMPT)
# 4. Run the agent
question = "What is the population of Paris, and what is the square root of that number?"
response = agent.run(question)

print("Answer:", response)

