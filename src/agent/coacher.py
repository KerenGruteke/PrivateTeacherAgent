from datetime import datetime
from functools import cached_property

from src.agent.prompts import COACHER_SYSTEM_PROMPT, COACHER_USER_PROMPT
from src.utils.constants import CHAT_DEPLOYMENT_NAME, AZURE_OPENAI_ENDPOINT, API_VERSION
from src.utils.LLM_utils import SystemMessage, HumanMessage
from src.utils.LLM_utils import LoggingAzureChatOpenAI
from src.utils.helper_function import json_parser
from src.data.index_and_search import get_db_object

@cached_property
def get_model():
    llm = LoggingAzureChatOpenAI(
        agent_name="COACHER",
        azure_deployment=CHAT_DEPLOYMENT_NAME,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        openai_api_version=API_VERSION,
        openai_api_type="azure",
        # temperature=0,
    )
    return llm


def get_coached_response(messages):
    messages.extend(
        [
            SystemMessage(
                content=COACHER_SYSTEM_PROMPT
            ),

            HumanMessage(
                content=COACHER_USER_PROMPT
            )
        ]
    )

    response = get_model.genearte(messages)
    return response.content
