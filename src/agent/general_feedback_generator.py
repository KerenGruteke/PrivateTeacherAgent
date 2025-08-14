from functools import cached_property

from src.agent.prompts import FINAL_FEEDBACK_SYSTEM_PROMPT, FINAL_FEEDBACK_USER_PROMPT
from src.utils.constants import CHAT_DEPLOYMENT_NAME, AZURE_OPENAI_ENDPOINT, API_VERSION
from src.utils.LLM_utils import SystemMessage, HumanMessage
from src.utils.LLM_utils import LoggingAzureChatOpenAI
from src.utils.helper_function import json_parser
from src.agent.student_evaluator import update_student_course_status


@lru_cache(maxsize=1)
def get_model():
    llm = LoggingAzureChatOpenAI(
        agent_name="FINAL_FEEDBACK",
        azure_deployment=CHAT_DEPLOYMENT_NAME,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        openai_api_version=API_VERSION,
        openai_api_type="azure",
        temperature=0,
    )
    return llm


def final_feedback(session_summary:str=None, student_id:str=None, course:str=None, **kwargs):
    """
    Generate supportive final feedback for a student session and update their course status.

    Uses an LLM to create personalized feedback based on the session summary, student ID,
    course, and topic. Feedback is printed and stored in the student database.

    Args:
        session_summary (str, optional): Summary of the learning session.
        student_id (str, optional): Student's unique identifier.
        course (str, optional): Course name or ID.

    Returns:
        dict: Parsed JSON output from the LLM with keys like "feedback" and "suggestions".
    """
    llm = get_model()

    messages = [
        SystemMessage(content=FINAL_FEEDBACK_SYSTEM_PROMPT),
        HumanMessage(content=FINAL_FEEDBACK_USER_PROMPT.format(
            course=course,
            session_summary=session_summary
        ))
    ]

    response = llm(messages)
    feedback_session = response.content
    print(feedback_session)
    update_student_course_status(student_id, course, feedback_session)

    return feedback_session


if __name__ == "__main__":
    session_summary = (
        "During the session, the student practiced solving quadratic equations using the quadratic formula. "
        "They improved in remembering all parts of the solution and in simplifying square roots. "
        "Overall, their problem-solving approach became more consistent over the exercises."
    )
    student_id = "s001"
    course = "Math"
    topic = "Quadratic Equations"

    # Call the final_feedback tool
    feedback_output = final_feedback(
        session_summary=session_summary,
        student_id=student_id,
        course=course,
    )

    # Print the output
    print("\n--- Final Feedback ---")
    print(feedback_output)
