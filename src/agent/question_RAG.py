from functools import lru_cache

from langchain.tools import DuckDuckGoSearchRun
from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType

from src.agent.prompts import (
    GET_QUERY_TO_SEARCH_SYSTEM_PROMPT,
    GET_QUERY_TO_SEARCH_USER_PROMPT,
    GENERATE_QUESTION_SYSTEM_PROMPT,
    GENERATE_QUESTION_USER_PROMPT,
    INFER_SUBJECT_SYSTEM_PROMPT,
    INFER_SUBJECT_USER_PROMPT,
    SUBJECT_GUIDELINES_PROMPTS_DICT,
)
from src.utils.constants import CHAT_DEPLOYMENT_NAME, AZURE_OPENAI_ENDPOINT, API_VERSION
from src.utils.LLM_utils import SystemMessage, HumanMessage
from src.utils.LLM_utils import LoggingAzureChatOpenAI
from src.utils.helper_function import json_parser
from src.data.index_and_search import get_db_object, SUBJECT_TO_COLLECTION_NAME


# -------------------------------------
# LLM factory (cached; temperature=0 for stability)
# -------------------------------------
@lru_cache(maxsize=1)
def get_model() -> LoggingAzureChatOpenAI:
    return LoggingAzureChatOpenAI(
        agent_name="GENERATE_QUESTION",
        azure_deployment=CHAT_DEPLOYMENT_NAME,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        openai_api_version=API_VERSION,
        openai_api_type="azure",
        temperature=0,
    )


# -------------------------------------
# Small helpers
# -------------------------------------

def _call(llm: LoggingAzureChatOpenAI, sys: str, user: str) -> str:
    """
    Minimal wrapper to keep the call sites clean.
    Assumes llm([...]) returns an object with .content (adjust if your API differs).
    """
    messages = [
        SystemMessage(content=sys),
        HumanMessage(content=user),
    ]
    resp = llm(messages)
    # If your class returns a dict/string, adapt here:
    return getattr(resp, "content", str(resp))

def _remove_redundant_data_fields(docs):
    if 'question_description' in docs:
        # remove question description
        [doc.pop('question_description') for doc in docs]
    return docs

# -------------------------------------
# Subject inference
# -------------------------------------
def infer_subject_from_request(request: str) -> str:
    llm = get_model()
    subject = _call(
        llm,
        INFER_SUBJECT_SYSTEM_PROMPT,
        INFER_SUBJECT_USER_PROMPT.format(request=request),
    ).strip()

    # Validate / fallback (TODO: consider retry if invalid)
    if subject not in SUBJECT_GUIDELINES_PROMPTS_DICT:
        subject = "SAT"  # default_subject
    return subject


# -------------------------------------
# Build structured query for the chosen subject
# -------------------------------------
def get_query_to_search(request: str, subject: str) -> str:
    llm = get_model()
    response = _call(
        llm,
        GET_QUERY_TO_SEARCH_SYSTEM_PROMPT,
        GET_QUERY_TO_SEARCH_USER_PROMPT.format(
            request=request,
            subject=subject,
            subject_guidelines=SUBJECT_GUIDELINES_PROMPTS_DICT[subject],
        ),
    )
    return response


# -------------------------------------
# DB search tool (must be used first by the agent)
# -------------------------------------
def search_in_DB(request: str) -> str:
    """
    Search for relevant documents in the database based on the user request.
    Returns:
        JSON string: [{"doc_id": "...", "snippet": "...", "metadata": {...}}, ...]
    """
    subject = infer_subject_from_request(request)
    collection_name = SUBJECT_TO_COLLECTION_NAME[subject]
    query_to_search = get_query_to_search(request, subject)

    docs = get_db_object().search_by_query_vec(
        collection_name=collection_name,
        query=query_to_search,
        top_k=2,
    )
    docs = _remove_redundant_data_fields(docs)
    return str(docs)


# -------------------------------------
# Web search tool (fallback)
# -------------------------------------
_ddg = DuckDuckGoSearchRun()
tools = [
    Tool(
        name="Search_in_DB",
        func=search_in_DB,
        description=("Use this FIRST (exactly once). Retrieves candidate materials from our internal DB "
                    "as a JSON string of short snippets. Prefer questions derived from these snippets.")
    ),
    Tool(
        name="Search_in_Web",
        func=_ddg.run,
        description="Use ONLY if DB results are insufficient or off-topic. Call at most once."
    ),
]

# -------------------------------------
# Agent entry point
# -------------------------------------
def generate_question_agent(request: str) -> str:
    """
    Runs a ReAct agent to generate a question based on the user request. It uses an Search_in_DB tool
    and may also use an external web search tool if necessary.

    Input: User request string
    Example: "Can you give me a question about photosynthesis?"

    Output: a single JSON object
    Example:
    {
        "subject": "...",
        "question": "...",
        "solution": "...",
        "hint": "...(optional)",
        "difficulty": "easy|medium|hard (optional)",
        "source": "db|web",
        "provenance": "doc ids or urls"
    }
    """
    llm = get_model()

    # ReAct agents expect a single prompt string; we concatenate system + user parts.
    prompt_text = (
        GENERATE_QUESTION_SYSTEM_PROMPT.strip()
        + "\n\n"
        + GENERATE_QUESTION_USER_PROMPT.format(request=request).strip()
    )

    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        max_iterations=4,                 # hard stop to avoid loops
        early_stopping_method="generate", # end gracefully if unsure
        handle_parsing_errors=True,       # more resilient formatting
        # return_intermediate_steps=True,  # <- enable while debugging
    )

    # agent.run returns a string; our prompt enforces JSON-only output.
    response = agent.run(prompt_text)
    return response


if __name__ == "__main__":
    # test the agent
    test_request = "Can you give me a question about photosynthesis?"
    response = generate_question_agent(test_request)
    print("Agent response:")
    print(response)

    test_request = "Can you give me a hard question in physics?"
    response = generate_question_agent(test_request)
    print("Agent response:")
    print(response)
