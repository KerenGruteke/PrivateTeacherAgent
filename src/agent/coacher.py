# src/agent/coacher.py
from functools import lru_cache
from pydantic import BaseModel, Field
from langchain.tools import StructuredTool
from src.agent.prompts import COACHER_SYSTEM_PROMPT, COACHER_USER_PROMPT
from src.utils.LLM_utils import SystemMessage, HumanMessage, LoggingAzureChatOpenAI
from src.utils.constants import CHAT_DEPLOYMENT_NAME, AZURE_OPENAI_ENDPOINT, API_VERSION


@lru_cache(maxsize=1)
def get_model() -> LoggingAzureChatOpenAI:
    """
    Return a cached Chat LLM for the Coacher tool (temperature=0 for stable phrasing).
    """
    return LoggingAzureChatOpenAI(
        agent_name="COACHER",
        azure_deployment=CHAT_DEPLOYMENT_NAME,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        openai_api_version=API_VERSION,
        openai_api_type="azure",
        temperature=0,
    )


def get_coacher_response(student_state: str) -> str:
    """
    Prints a short, student-facing motivational message tailored to the student's state.

    Parameters
    ----------
    student_state : str
        A concise summary from the main Private Teacher agent describing the learner’s
        current situation (e.g., focus areas, recent progress/mistakes, confidence level).

    Returns
    -------
    str
        Approval that the coacher message was printed.
    """
    messages = [
        SystemMessage(content=COACHER_SYSTEM_PROMPT),
        HumanMessage(content=COACHER_USER_PROMPT.format(student_state=student_state)),
    ]
    resp = get_model()(messages)
    message = getattr(resp, "content", str(resp)).strip()
    print(f"\n\n💖🤖 AI Teacher: {message}")
    return "coacher message was printed, now proceed to another question or continue guiding the student solving the current question if needed"


if __name__ == "__main__":
    student_state = "Struggling with algebra"
    message = get_coacher_response(student_state)
    print(message)