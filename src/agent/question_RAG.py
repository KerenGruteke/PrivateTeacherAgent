from functools import cached_property
from langchain.tools import DuckDuckGoSearchRun
from langchain.agents import initialize_agent, Tool
from src.agent.prompts import GET_QUERY_TO_SEARCH_SYSTEM_PROMPT, GET_QUERY_TO_SEARCH_USER_PROMPT, \
    REWRITE_QUESTION_SYSTEM_PROMPT, REWRITE_QUESTION_USER_PROMPT, \
    GENERATE_QUESTION_SYSTEM_PROMPT, GENERATE_QUESTION_USER_PROMPT, \
    INFER_SUBJECT_SYSTEM_PROMPT, INFER_SUBJECT_USER_PROMPT, SUBJECT_GUIDELINES_PROMPTS_DICT
from src.utils.constants import CHAT_DEPLOYMENT_NAME, AZURE_OPENAI_ENDPOINT, API_VERSION
from src.utils.LLM_utils import SystemMessage, HumanMessage
from src.utils.LLM_utils import LoggingAzureChatOpenAI
from src.utils.helper_function import json_parser
from src.data.index_and_search import get_db_object
from langchain.agents.agent_types import AgentType
from src.data.index_and_search import SUBJECT_TO_COLLECTION_NAME


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


def infer_subject_from_request(request):
    # call the llm API using get_query_to_search_prompt
    messages = get_model.generate(
        [
            SystemMessage(
                content=INFER_SUBJECT_SYSTEM_PROMPT
            ),

            HumanMessage(
                content=INFER_SUBJECT_USER_PROMPT.format(
                request=request,
            ))
        ]
    )

    response = get_model.generate(messages)
    subject = response.content.strip()
    
    # TODO: change logic here - maybe retry
    # validate that subject is one of Science / Math / History / SAT
    if subject not in SUBJECT_GUIDELINES_PROMPTS_DICT:
        subject = 'SAT' # default_subject
    return subject


def get_query_to_search(request, subject):
    # call the llm API using get_query_to_search_prompt
    messages = get_model.generate(
        [
            SystemMessage(
                content=GET_QUERY_TO_SEARCH_SYSTEM_PROMPT
            ),

            HumanMessage(
                content=GET_QUERY_TO_SEARCH_USER_PROMPT.format(
                request=request,
                subject=subject,
                subject_guidelines=SUBJECT_GUIDELINES_PROMPTS_DICT[subject]
            ))
        ]
    )

    response = get_model.generate(messages)
    json_response = json_parser(response.content)
    return json_response["query_to_search"]


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

def search_in_DB(request: str)->str:
    """
    Search for relevant documents in the database based on the user request.

    Args:
        request (str): The user request to search for.

    Returns:
        string with the retrieved documents #TODO:validate and make it more clear
    """
    subject = infer_subject_from_request(request)
    collection_name = SUBJECT_TO_COLLECTION_NAME[subject]
    query_to_search = get_query_to_search(request, subject)
    retrieved_docs = get_db_object().search_by_query_vec(collection_name, query_to_search, top_k=5)
    return retrieved_docs

# Define tools
search = DuckDuckGoSearchRun()
tools = [
    Tool(
        name="Search_in_DB",
        func=search_in_DB,
        description="You must use this tool first to get optional questions."
    ),
    Tool(
        name="Search_in_Web",
        func=search.run,
        description="Useful for searching the web for another question in case the Search_in_DB wasn't useful."
    ),
]

def generate_question_agent(request):
    llm = get_model()
    
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
    )
    
    # The idea is that the agent can search for more infromation using the 
    # search tool to have a new question if the question that is based on 
    # RAG is not good enough
    messages = (
        [
            SystemMessage(
                content=GENERATE_QUESTION_SYSTEM_PROMPT
            ),

            HumanMessage(
                content=GENERATE_QUESTION_USER_PROMPT.format(
                    request=request,
                )
            )
        ]
    )

    response = agent.run(messages)
    return response.content