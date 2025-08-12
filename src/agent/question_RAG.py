from functools import cached_property

from src.agent.prompts import GET_QUERY_TO_SEARCH_SYSTEM_PROMPT, GET_QUERY_TO_SEARCH_USER_PROMPT, \
    REWRITE_QUESTION_SYSTEM_PROMPT, REWRITE_QUESTION_USER_PROMPT
from src.utils.constants import CHAT_DEPLOYMENT_NAME, AZURE_OPENAI_ENDPOINT, API_VERSION
from src.utils.LLM_utils import SystemMessage, HumanMessage
from src.utils.LLM_utils import LoggingAzureChatOpenAI
from src.utils.helper_function import json_parser
from src.data.index_and_search import get_db_object


@cached_property
def get_model():
    llm = LoggingAzureChatOpenAI(
        agent_name="GENERATE_QUESTION",
        azure_deployment=CHAT_DEPLOYMENT_NAME,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        openai_api_version=API_VERSION,
        openai_api_type="azure",
        # temperature=0,
    )
    return llm


def get_query_to_search(user_request):
    # call the llm API using get_query_to_search_prompt
    messages = get_model.generate(
        [
            SystemMessage(
                content=GET_QUERY_TO_SEARCH_SYSTEM_PROMPT
            ),

            HumanMessage(
                content=GET_QUERY_TO_SEARCH_USER_PROMPT.format(
                user_request=user_request,
            ))
        ]
    )

    response = get_model.genearte(messages)
    json_response = json_parser(response.content)
    return json_response["query_to_search"], json_response["collection_name"]


def rewrite_question(user_request, retrieved_docs):
    messages = get_model.generate(
        [
            SystemMessage(
                content=REWRITE_QUESTION_SYSTEM_PROMPT
            ),

            HumanMessage(
                content=REWRITE_QUESTION_USER_PROMPT.format(
                    user_request=user_request,
                    retrieved_docs=retrieved_docs
                ))
        ]
    )

    rewritten_question = get_model.genearte(messages).response
    return rewritten_question


def generate_question(user_request):
    query_to_search, collection_name = get_query_to_search(user_request)
    retrieved_docs = get_db_object().search_by_query_vec(collection_name, query_to_search, top_k=5)
    rewritten_question = rewrite_question(user_request, retrieved_docs)
    return rewritten_question



