from datetime import datetime
from functools import cached_property

from src.agent.prompts import UPDATE_STUDENT_STATUS_SYSTEM_PROMPT, UPDATE_STUDENT_STATUS_USER_PROMPT
from src.utils.constants import CHAT_DEPLOYMENT_NAME, AZURE_OPENAI_ENDPOINT, API_VERSION
from src.utils.LLM_utils import SystemMessage, HumanMessage
from src.utils.LLM_utils import LoggingAzureChatOpenAI
from src.utils.helper_function import json_parser
from src.data.index_and_search import get_db_object


@cached_property
def get_model():
    llm = LoggingAzureChatOpenAI(
        agent_name="STUDENT_EVALUATOR",
        azure_deployment=CHAT_DEPLOYMENT_NAME,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        openai_api_version=API_VERSION,
        openai_api_type="azure",
        # temperature=0,
    )
    return llm


def get_status(student_id, topic):
    student_data_dict = get_db_object().get_items_data(
        collection_name="students_db",
        item_ids=[student_id],
        id_col="student_id",
    )

    return [topic_status for topic_status in student_data_dict[student_id]["status"] if topic in topic_status]



def update_student_level(student_id, topic, current_session_feedback):
    student_data_dict = get_db_object().get_items_data(
        collection_name="students_db",
        item_ids=[student_id],
        id_col="student_id",
    )

    # call the llm API using get_query_to_search_prompt
    messages = get_model.generate(
        [
            SystemMessage(
                content=UPDATE_STUDENT_STATUS_SYSTEM_PROMPT
            ),

            HumanMessage(
                content=UPDATE_STUDENT_STATUS_USER_PROMPT.format(
                current_session_feedback=current_session_feedback,
            ))
        ]
    )

    response = get_model.genearte(messages)
    parsed_results = json_parser(response)
    parsed_results.update({"date": datetime.now().date()})
    student_data_dict[student_id]["status"][topic].append(parsed_results)


    get_db_object().update_metadata(
        collection_name="students_db",
        item_id=student_id,
        new_metadata=student_data_dict
    )

    #     (
    #         [0.0],
    #         {
    #             "student_id": "S002",
    #             "name": "Bob",
    #             "status": {
    #                 "math": [{"score": 60, "note": "needs more algebra", "date": "18-07-2025"},
    #                         {"score": 90, "note": "should practice multiplication", "date": "18-07-2025"}],
    #                 "history": [{"score": 95, "note": "Practice Renaissance", "date": "10-06-2024"}]
    #             }
    #         }
    #     ),
    # ]