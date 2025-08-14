from dotenv import load_dotenv
from functools import lru_cache
from src.utils.constants import CHAT_DEPLOYMENT_NAME, AZURE_OPENAI_ENDPOINT, API_VERSION, DEBUG_MODE
from src.utils.LLM_utils import LoggingAzureChatOpenAI

from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
from src.agent.coacher import get_coacher_response
from src.agent.question_RAG import generate_question_agent
from src.agent.answer_evaluator import evaluate_answer
from src.agent.hand_in_hand_solver import hand_in_hand_agent
from src.agent.prompts import (
    INITIALIZE_MAIN_PRIVATE_TEACHER_SYSTEM_PROMPT,
    INITIALIZE_MAIN_PRIVATE_TEACHER_USER_PROMPT
)
from src.utils.user_response import present_message_to_user
from src.agent.student_evaluator import get_student_course_status
from src.agent.general_feedback_generator import provide_final_feedback

load_dotenv()

# -------------------------------------
# LLM factory (cached; temperature=0 for stability)
# -------------------------------------
@lru_cache(maxsize=1)
def get_model() -> LoggingAzureChatOpenAI:
    return LoggingAzureChatOpenAI(
        agent_name="MAIN_PRIVATE_AGENT",
        azure_deployment=CHAT_DEPLOYMENT_NAME,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        openai_api_version=API_VERSION,
        openai_api_type="azure",
        temperature=0,
    )

# Define tools
tools = [
    Tool(
        name="generate_question_agent",
        func=generate_question_agent,
        description=(
            "Generate a practice question and its solution based on the user's request. "
            "First searches the internal subject-specific database for relevant material; "
            "if needed, can also search the web for additional context. "
            "Outputs a single JSON object with the question, correct solution, and optional hint/difficulty."
        )
    ),
    Tool(
        name="present_message_to_user",
        func=present_message_to_user,
        description=("Present a message to the student and get their reply."),
    ),
    Tool(
        name="evaluate_answer",
        func=evaluate_answer,
        description=(
            "Evaluate a student's answer against the reference solution. "
            "Returns JSON with correctness, score (0–1), feedback, common_mistakes. "
            "Also logs/merges common mistakes into the 'common_mistakes' DB for future use."
        ),
    ),
    Tool(
        name="hand_in_hand_agent",
        func=hand_in_hand_agent,
        description=(
            "An interactive tutoring tool that guides a student through a question step-by-step. "
            "It breaks the question into smaller sub-steps, evaluates the student's answers, "
            "and provides constructive, personalized feedback to help the student improve."
        ),
    ),
    Tool(
        name="get_coacher_response",
        func=get_coacher_response,
        description=(
        "Produce and a short, student-friendly motivational message based on the provided student_state. "
        "Use this to encourage the learner, explain the value of the current course, or suggest a tiny next step. "
        "After using this tool proceed with the student interaction."
        ),
    ),
    Tool(
        name="provide_final_feedback",
        func=provide_final_feedback,
        description=(
            "Generate supportive final feedback for a student session "
            "which is based on a session_summary that you give."
            "Use this tool in the end of the lesson."
        )
    )
    
]

# Create ReAct agent
def init_private_teacher(student_id, course, user_message):
    student_evaluation_notes = get_student_course_status(student_id, course)

    agent = initialize_agent(
        tools=tools,
        llm=get_model(),
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=DEBUG_MODE,
        handle_parsing_errors=True,       # more resilient formatting
    )

    prompt = INITIALIZE_MAIN_PRIVATE_TEACHER_SYSTEM_PROMPT + "\n" + \
            INITIALIZE_MAIN_PRIVATE_TEACHER_USER_PROMPT.format(
            student_id=student_id,
            course=course,
            user_message=user_message,
            student_evaluation_notes=student_evaluation_notes
        )
    
    return agent, prompt
