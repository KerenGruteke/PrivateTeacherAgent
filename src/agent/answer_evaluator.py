from functools import lru_cache
import json
from typing import List, Dict
import pandas as pd
from src.agent.prompts import EVALUATE_ANSWER_SYSTEM_PROMPT, EVALUATE_ANSWER_USER_PROMPT
from src.utils.constants import CHAT_DEPLOYMENT_NAME, AZURE_OPENAI_ENDPOINT, API_VERSION
from src.utils.LLM_utils import SystemMessage, HumanMessage, LoggingAzureChatOpenAI
from src.utils.helper_function import json_parser
from src.data.index_and_search import get_db_object, index_df


COMMON_MISTAKES_COLLECTION = "common_mistakes"
COMMON_MISTAKES_ID_COL = "question_description"  # used as the unique ID in this collection


@lru_cache(maxsize=1)
def get_model() -> LoggingAzureChatOpenAI:
    """Return a cached Chat LLM for the Answer Evaluator (temperature=0 for determinism)."""
    return LoggingAzureChatOpenAI(
        agent_name="ANSWER_EVALUATOR",
        azure_deployment=CHAT_DEPLOYMENT_NAME,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        openai_api_version=API_VERSION,
        openai_api_type="azure",
        temperature=0,
    )


def _dedupe(seq: List[str]) -> List[str]:
    """Order‑preserving de-duplication of a list of strings."""
    return list(dict.fromkeys([s for s in seq if isinstance(s, str) and s.strip()]))


def evaluate_answer(student_answer: str, solution: str, question: str, topic: str) -> str:
    """
    Evaluate a student's answer against the reference solution and (optionally) log common mistakes.

    Parameters
    ----------
    student_answer : str
        The student's response.
    solution : str
        The reference solution or marking guide.
    question : str
        The question text.
    topic : str
        The subject/topic label (e.g., 'Math', 'Science', 'History', 'SAT').

    Returns
    -------
    str
        JSON string with evaluation fields
        (e.g., correctness, score, feedback, common_mistakes).
    """
    llm = get_model()
    messages = [
        SystemMessage(content=EVALUATE_ANSWER_SYSTEM_PROMPT),
        HumanMessage(
            content=EVALUATE_ANSWER_USER_PROMPT.format(
                topic=topic,
                question=question,
                solution=solution,
                student_answer=student_answer,
            )
        ),
    ]

    resp = llm(messages)
    text = getattr(resp, "content", str(resp)).strip()
    output = json_parser(text)  # must return a dict; raise on invalid JSON

    # ---- Persist/merge common mistakes in the DB ----
    qid = f"Topic: {topic}\nQuestion: {question}"
    new_mistakes: List[str] = _dedupe(output.get("common_mistakes", []) or [])

    if new_mistakes:
        # Try to fetch an existing record for this question
        existing = get_db_object().get_items_data(
            collection_name=COMMON_MISTAKES_COLLECTION,
            item_ids=[qid],
            id_col=COMMON_MISTAKES_ID_COL,
        )

        if not existing:
            # Create a new record
            df = pd.DataFrame(
                [{
                    COMMON_MISTAKES_ID_COL: qid,
                    "topic": topic,
                    "question": question,
                    "common_mistakes": new_mistakes,
                }]
            )
            index_df(
                df=df,
                index_by_col=COMMON_MISTAKES_ID_COL,
                need_to_embed_col=True,
                id_col=COMMON_MISTAKES_ID_COL,
                collection_name=COMMON_MISTAKES_COLLECTION,
            )
        else:
            # Merge into the existing record
            payload_map: Dict[str, Dict] = existing  # {qid: payload}
            record = payload_map.get(qid, {}) or {}
            merged = _dedupe((record.get("common_mistakes") or []) + new_mistakes)
            record.update({
                COMMON_MISTAKES_ID_COL: qid,
                "topic": topic,
                "question": question,
                "common_mistakes": merged,
            })
            # Update only this point's payload
            get_db_object().update_metadata(
                collection_name=COMMON_MISTAKES_COLLECTION,
                item_id=qid,
                new_metadata=record,
            )

    return json.dumps(output, ensure_ascii=False)


# ------------
# Init
# ------------

def init_mistakes_DB_with_few_examples() :

    # Prepare example data
    data = [
        # Math
        {
            "topic": "Math",
            "question_description": "Topic: math\nQuestion: What is the result of 7 + 7?",
            "question": "What is the result of 7 + 7?",
            "common_mistakes": [
                "Answering 15 due to a simple addition error",
                "Confusing addition with multiplication and answering 49",
                "Writing '77' because of concatenation instead of addition"
            ]
        },
        {
            "topic": "Math",
            "question_description": "Topic: math\nQuestion: What is the area of a square with side length 5 cm?",
            "question": "What is the area of a square with side length 5 cm?",
            "common_mistakes": [
                "Multiplying by 4 instead of squaring the side (answering 20)",
                "Using wrong units (writing '5 cm²' instead of 25 cm²)"
            ]
        },

        # Science
        {
            "topic": "Science",
            "question_description": "Topic: science\nQuestion: What is the boiling point of water at sea level in Celsius?",
            "question": "What is the boiling point of water at sea level in Celsius?",
            "common_mistakes": [
                "Answering 100°F instead of 100°C",
                "Confusing boiling point with melting point and answering 0°C",
                "Saying 'it depends' without specifying standard atmospheric pressure"
            ]
        },
        {
            "topic": "Science",
            "question_description": "Topic: science\nQuestion: What gas is released during photosynthesis?",
            "question": "What gas is released during photosynthesis?",
            "common_mistakes": [
                "Answering carbon dioxide instead of oxygen",
                "Saying 'nitrogen' because it is abundant in air",
                "Confusing with respiration and mentioning 'water vapor'"
            ]
        },

        # SAT
        {
            "topic": "SAT",
            "question_description": "Topic: SAT US History\nQuestion: In what year was the U.S. Constitution ratified?",
            "question": "In what year was the U.S. Constitution ratified?",
            "common_mistakes": [
                "Answering 1776 (year of Declaration of Independence)",
                "Answering 1781 (year Articles of Confederation were ratified)",
                "Answering 1791 (year Bill of Rights was adopted)"
            ]
        },
        {
            "topic": "SAT",
            "question_description": "Topic: SAT World History\nQuestion: In which years did World War II begin and end?",
            "question": "In which years did World War II begin and end?",
            "common_mistakes": [
                "Answering 1914–1918 (confusing with WWI)",
                "Answering 1939–1944 (forgetting the war ended in 1945)",
                "Answering 1941–1945 (thinking it started with U.S. entry)"
            ]
        },

        # History
        {
            "topic": "History",
            "question_description": "Topic: history\nQuestion: In what year did Martin Luther King Jr. deliver his 'I Have a Dream' speech?",
            "question": "In what year did Martin Luther King Jr. deliver his 'I Have a Dream' speech?",
            "common_mistakes": [
                "Answering 1960 (confusing with earlier civil rights events)",
                "Answering 1968 (year of MLK's assassination)",
                "Answering 1955 (confusing with Montgomery Bus Boycott)"
            ]
        },
        {
            "topic": "History",
            "question_description": "Topic: history\nQuestion: What year was the first human moon landing?",
            "question": "What year was the first human moon landing?",
            "common_mistakes": [
                "Answering 1968 (confusing with Apollo 8 orbit)",
                "Answering 1970 (confusing with Apollo 13 incident)",
                "Answering 1957 (confusing with Sputnik launch)"
            ]
        }
    ]

    # Convert to DataFrame
    df = pd.DataFrame(data)

    # Index into Qdrant using your index_df function
    index_df(
        df=df,
        index_by_col="question_description",  # column to embed
        need_to_embed_col=True,               # embed for semantic search
        id_col=COMMON_MISTAKES_ID_COL,
        collection_name=COMMON_MISTAKES_COLLECTION    # Qdrant collection name
    )

    print(f"Indexed {len(df)} common mistake examples into '{COMMON_MISTAKES_COLLECTION}' collection.")


import time
import pandas as pd
from typing import List, Dict, Any

from src.data.index_and_search import get_db_object, index_df
from src.agent.answer_evaluator import evaluate_answer  # adjust import if needed

COMMON_MISTAKES_COLLECTION = "common_mistakes"
COMMON_MISTAKES_ID_COL = "question_description"


# -----------------------
# Small DB helper utils
# -----------------------
def _collection_size(collection_name: str) -> int:
    """Return number of points in a Qdrant collection."""
    db = get_db_object()
    res = db.qdrant_client.count(collection_name=collection_name)
    # qdrant-client returns CountResult(count=<int>)
    return int(getattr(res, "count", res))


def _fetch_record(qid: str) -> Dict[str, Any] | None:
    """Retrieve a single payload by question_description (qid)."""
    db = get_db_object()
    res = db.get_items_data(
        collection_name=COMMON_MISTAKES_COLLECTION,
        item_ids=[qid],
        id_col=COMMON_MISTAKES_ID_COL,
    )
    return None if not res else res.get(qid)


def _ensure_existing_record(qid: str, topic: str, question: str, mistakes: List[str]) -> None:
    """Create a baseline record if it doesn't exist."""
    if _fetch_record(qid) is None:
        df = pd.DataFrame([{
            COMMON_MISTAKES_ID_COL: qid,
            "topic": topic,
            "question": question,
            "common_mistakes": mistakes,
        }])
        index_df(
            df=df,
            index_by_col=COMMON_MISTAKES_ID_COL,
            need_to_embed_col=True,
            id_col=COMMON_MISTAKES_ID_COL,
            collection_name=COMMON_MISTAKES_COLLECTION,
        )


# -------------------------------------------------------
# Tests
# -------------------------------------------------------
def _test_common_mistake_for_new_question():
    """
    Create a unique new question, run evaluate_answer with an intentionally wrong answer,
    and assert:
        1) collection size increases by 1
        2) new record has at least one common mistake
    """
    topic = "Math"
    # Make the question unique to avoid collisions in repeated runs
    uniq = int(time.time())
    question = f"What is 12 + 7? (run={uniq})"
    solution = "The sum is 19."
    student_answer = "21"  # wrong on purpose
    qid = f"Topic: {topic}\nQuestion: {question}"

    size_before = _collection_size(COMMON_MISTAKES_COLLECTION)

    # Run evaluation (this will insert the record)
    _ = evaluate_answer(
        student_answer=student_answer,
        solution=solution,
        question=question,
        topic=topic,
    )

    size_after = _collection_size(COMMON_MISTAKES_COLLECTION)
    assert size_after == size_before + 1, \
        f"Expected collection size to grow by 1 (before={size_before}, after={size_after})."

    record = _fetch_record(qid)
    assert record is not None, "New record not found after evaluate_answer."
    mistakes = record.get("common_mistakes", [])
    assert isinstance(mistakes, list) and len(mistakes) >= 1, \
        f"Expected at least one common mistake, got: {mistakes}"

    print("✅ _test_common_mistake_for_new_question passed.")


def _test_common_mistake_for_existing_question():
    """
    Ensure an existing record is updated:
        1) collection size stays the same
        2) number of common mistakes for this question increases
    """
    topic = "Science"
    question = "What gas is released during photosynthesis?"
    solution = "Oxygen."
    qid = f"Topic: {topic}\nQuestion: {question}"

    # Ensure baseline record exists with at least one mistake
    baseline_mistakes = ["Answering carbon dioxide instead of oxygen"]
    _ensure_existing_record(qid, topic, question, baseline_mistakes)

    size_before = _collection_size(COMMON_MISTAKES_COLLECTION)
    record_before = _fetch_record(qid)
    assert record_before is not None, "Baseline record missing unexpectedly."
    before_count = len(record_before.get("common_mistakes", []))

    # Intentionally wrong/confused answer to elicit a *new* mistake if possible
    student_answer = "Nitrogen"
    _ = evaluate_answer(
        student_answer=student_answer,
        solution=solution,
        question=question,
        topic=topic,
    )

    size_after = _collection_size(COMMON_MISTAKES_COLLECTION)
    record_after = _fetch_record(qid)
    after_count = len(record_after.get("common_mistakes", [])) if record_after else 0

    assert size_after == size_before, \
        f"Collection size should not change for existing record (before={size_before}, after={size_after})."
    assert after_count > before_count, \
        f"Expected more common mistakes after update (before={before_count}, after={after_count}). " \
        f"If equal, the LLM likely returned duplicate mistakes."

    print("✅ _test_common_mistake_for_existing_question passed.")


if __name__ == "__main__":
    # init_mistakes_DB_with_few_examples()
    _test_common_mistake_for_new_question()
    _test_common_mistake_for_existing_question()
