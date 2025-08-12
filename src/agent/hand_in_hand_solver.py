from functools import cached_property

from src.agent.prompts import EVALUATE_ANSWER_SYSTEM_PROMPT, EVALUATE_ANSWER_USER_PROMPT
from src.utils.constants import CHAT_DEPLOYMENT_NAME, AZURE_OPENAI_ENDPOINT, API_VERSION
from src.utils.LLM_utils import SystemMessage, HumanMessage
from src.utils.LLM_utils import LoggingAzureChatOpenAI, get_embedding_object
from src.utils.helper_function import json_parser
from src.data.index_and_search import get_db_object
from langchain.agents import initialize_agent, Tool
from langchain.tools import DuckDuckGoSearchRun
from langchain.agents.agent_types import AgentType
from src.agent.answer_evaluator import evaluate_answer


@cached_property
def get_model():
    llm = LoggingAzureChatOpenAI(
        agent_name="HAND_IN_HANS",
        azure_deployment=CHAT_DEPLOYMENT_NAME,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        openai_api_version=API_VERSION,
        openai_api_type="azure",
        temperature=0,
    )
    return llm


def get_student_answer(question):
     return str(input(question))

def get_common_mistakes(topic, question):
    qid = f"Topic: {topic}\nQuestion: {question}"

    relevant_mistakes = get_db_object().search_by_query_vec(
        "common_mistakes",
        query=qid,
        top_k=3
    )

    common_mistakes = [mistake for q_mistakes in relevant_mistakes for mistake in q_mistakes["common mistakes"]]
    return common_mistakes


# 2. Define tools
tools = [
    Tool(
        name="Get Student Answer",
        func=get_student_answer,
        description="Get Student Answer For Current Sub Question"  # TODO: Add description
    ),
    Tool(
        name="Evaluate Student Answer",
        func=evaluate_answer,
        description="" #TODO: Add description
    ),
    Tool(
        name="Get Common Mistakes",
        func=get_common_mistakes,
        description="You can search for common mistakes in similar question to better analyze where students struggles"
    )
]

def hand_in_hand_agent(topic, question, solution, student_answer):
    llm = get_model()

    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, #TODO: Check what is and how is affect
        verbose=True,
    )

    messages = (
        [
            SystemMessage(
                content=EVALUATE_ANSWER_SYSTEM_PROMPT
            ),

            HumanMessage(
                content=EVALUATE_ANSWER_USER_PROMPT.format(
                    topic=topic,
                    question=question,
                    solution=solution,
                    student_answer=student_answer,
                )
            )
        ]
    )

    response = agent.run(messages)
    return response.content