from functools import cached_property

from src.agent.prompts import FINAL_FEEDBACK_SYSTEM_PROMPT, FINAL_FEEDBACK_USER_PROMPT
from src.utils.constants import CHAT_DEPLOYMENT_NAME, AZURE_OPENAI_ENDPOINT, API_VERSION
from src.utils.LLM_utils import SystemMessage, HumanMessage
from src.utils.LLM_utils import LoggingAzureChatOpenAI
from src.utils.helper_function import json_parser
from src.data.index_and_search import get_db_object
from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
from src.agent.student_evaluator import update_student_course_status

@cached_property
def get_model():
    llm = LoggingAzureChatOpenAI(
        agent_name="FINAL_FEEDBACK",
        azure_deployment=CHAT_DEPLOYMENT_NAME,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        openai_api_version=API_VERSION,
        openai_api_type="azure",
        temperature=0.7,
    )
    return llm


tools = [
        Tool(
        name="Update Student Course Status",
        func=update_student_course_status,
        description="Convert qualitative session feedback into a structured status update and store it in the student DB.",
    ),
]


def final_feedback(messages, student_id, course):
    messages.extend(
        [
            SystemMessage(
                content=FINAL_FEEDBACK_SYSTEM_PROMPT
            ),

            HumanMessage(
                content=FINAL_FEEDBACK_USER_PROMPT
            )
        ]
    )

    response = get_model.genearte(messages)
    json_output = json_parser(response.content)

    print(json_output["feedback"])
    update_student_level(student_id, course, json_output["feedback"])

    return response.content