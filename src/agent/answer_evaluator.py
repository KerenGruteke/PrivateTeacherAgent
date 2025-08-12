from functools import cached_property

import pandas as pd

from src.agent.prompts import EVALUATE_ANSWER_SYSTEM_PROMPT, EVALUATE_ANSWER_USER_PROMPT
from src.utils.constants import CHAT_DEPLOYMENT_NAME, AZURE_OPENAI_ENDPOINT, API_VERSION
from src.utils.LLM_utils import SystemMessage, HumanMessage
from src.utils.LLM_utils import LoggingAzureChatOpenAI
from src.utils.helper_function import json_parser
from src.data.index_and_search import get_db_object, index_df

@cached_property
def get_model():
    llm = LoggingAzureChatOpenAI(
        agent_name="ANSWER_EVALUATOR",
        azure_deployment=CHAT_DEPLOYMENT_NAME,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        openai_api_version=API_VERSION,
        openai_api_type="azure",
        temperature=0,
    )
    return llm


def evaluate_answer(student_answer, solution, question, topic):
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

    response = get_model.genearte(messages)
    output_json = json_parser(response.content)

    qid = f"Topic: {topic}\nQuestion: {question}"

    question_common_mistakes = get_db_object().get_items_data(
        collection_name="common_mistakes",
        item_ids=[qid],
        id_col="question_description",
    )

    if question_common_mistakes is None:
        df = pd.DataFrame(
            {
                "question_description": qid,
                "topic": topic,
                "question": question,
                "common mistakes": output_json['common mistakes'],
             }
        )
        index_df(
            df=df,
            index_by_col="question_description",
            need_to_embed_col=True,
            id_col="question_description",
            collection_name="common_mistakes",
        )
    else:
        payload_data = get_db_object().get_items_data(
            collection_name="common_mistakes",
            item_ids=[qid],
            id_col="question_description"
        )

        payload_data[qid]["common mistakes"].extend(output_json['common mistakes'])

        get_db_object().update_metadata(
            collection_name="common_mistakes",
            item_id=qid,
            new_metadata=payload_data
        )

    return response.content

    # Template for common mistake db
    # {
    #     "question_description": "Topic: math\nQuestion: what is the results of 7 + 7?",
    #     "topic": "math",
    #     "question": "what is the results of 7 + 7?",
    #     "common mistakes": ["bla", "bla"]
    # }