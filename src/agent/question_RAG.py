from functools import lru_cache

from langchain.tools import DuckDuckGoSearchRun
from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType

from src.agent.prompts import (
    GET_QUERY_TO_SEARCH_SYSTEM_PROMPT,
    GET_QUERY_TO_SEARCH_USER_PROMPT,
    GENERATE_QUESTION_SYSTEM_PROMPT,
    GENERATE_QUESTION_USER_PROMPT,
    INFER_course_SYSTEM_PROMPT,
    INFER_course_USER_PROMPT,
    course_GUIDELINES_PROMPTS_DICT,
)
from src.utils.constants import CHAT_DEPLOYMENT_NAME, AZURE_OPENAI_ENDPOINT, API_VERSION
from src.utils.LLM_utils import SystemMessage, HumanMessage
from src.utils.LLM_utils import LoggingAzureChatOpenAI
from src.utils.helper_function import json_parser
from src.data.index_and_search import get_db_object, COURSE_TO_COLLECTION_NAME
from src.agent.student_evaluator import get_student_course_status


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
# course inference
# -------------------------------------
def infer_course_from_request(request: str) -> str:
    llm = get_model()
    course = _call(
        llm,
        INFER_course_SYSTEM_PROMPT,
        INFER_course_USER_PROMPT.format(request=request),
    ).strip()

    # Validate / fallback (TODO: consider retry if invalid)
    if course not in course_GUIDELINES_PROMPTS_DICT:
        course = "SAT"  # default_course
    return course


# -------------------------------------
# Build structured query for the chosen course
# -------------------------------------
def get_query_to_search(request: str, course: str) -> str:
    llm = get_model()
    response = _call(
        llm,
        GET_QUERY_TO_SEARCH_SYSTEM_PROMPT,
        GET_QUERY_TO_SEARCH_USER_PROMPT.format(
            request=request,
            course=course,
            course_guidelines=course_GUIDELINES_PROMPTS_DICT[course],
        ),
    )
    return response


# -------------------------------------
# DB search tool (must be used first by the agent)
# -------------------------------------
def search_in_DB(request: str) -> str:
    """
    Search for relevant documents in the database based on the user request.
    Input: 
        - User request (str)
    Returns:
        JSON string: [{"doc_id": "...", "snippet": "...", "metadata": {...}}, ...]
    """
    course = infer_course_from_request(request)
    collection_name = COURSE_TO_COLLECTION_NAME[course]
    query_to_search = get_query_to_search(request, course)

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
def generate_question_agent(request: str, student_id: str) -> str:
    """
    Runs a ReAct agent to generate a question based on the user request. It uses an Search_in_DB tool
    and may also use an external web search tool if necessary.

    Input: 
    - User request (str)
    - Student ID (str)
    Example: "Can you give me a question about photosynthesis?"

    Output: a single JSON object
    Example:
    {
        "course": "...",
        "question": "...",
        "solution": "...",
        "hint": "...(optional)",
        "difficulty": "easy|medium|hard (optional)",
        "source": "db|web",
        "provenance": "doc ids or urls"
    }
    """
    llm = get_model()

    # Expand request by notes of student
    course = infer_course_from_request(request)
    student_evaluation_notes = get_student_course_status(student_id, course)
    request = f"Request:\n{request}\nEvaluation Notes:\n{student_evaluation_notes}"

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
    # test_request = "Can you give me a question about photosynthesis?"
    # response = generate_question_agent(test_request, student_id="S001")
    # print("Agent response:")
    # print(response)

    # test_request = "Can you give me a hard question in physics?"
    # response = generate_question_agent(test_request, student_id="S001")
    # print("Agent response:")
    # print(response)
    
    # Math
    test_request = "Can you give me a question about algebra?"
    response = generate_question_agent(test_request, student_id="S001")
    print("Agent response:")
    print(response)
