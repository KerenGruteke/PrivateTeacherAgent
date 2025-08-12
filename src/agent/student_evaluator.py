from datetime import datetime
from functools import lru_cache
from typing import Dict, Any
import pandas as pd

from src.agent.prompts import (
    UPDATE_STUDENT_STATUS_SYSTEM_PROMPT,
    UPDATE_STUDENT_STATUS_USER_PROMPT
)
from src.utils.constants import CHAT_DEPLOYMENT_NAME, AZURE_OPENAI_ENDPOINT, API_VERSION
from src.utils.LLM_utils import SystemMessage, HumanMessage, LoggingAzureChatOpenAI
from src.utils.helper_function import json_parser
from src.data.index_and_search import get_db_object, index_df

STUDENTS_COLLECTION = "students_db"


@lru_cache(maxsize=1)
def get_model() -> LoggingAzureChatOpenAI:
    """Return a cached LLM instance for Student Evaluator."""
    return LoggingAzureChatOpenAI(
        agent_name="STUDENT_EVALUATOR",
        azure_deployment=CHAT_DEPLOYMENT_NAME,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        openai_api_version=API_VERSION,
        openai_api_type="azure",
        temperature=0,
    )


# ----------------------
# Tool 1: Get status
# ----------------------
def get_student_course_status(student_id: str, course: str) -> Dict[str, Any]:
    """
    Retrieve the student's historical status entries for a given course.
    """
    student_data = get_db_object().get_items_data(
        collection_name="students_db",
        item_ids=[student_id],
        id_col="student_id",
    )
    if not student_data or student_id not in student_data:
        return {"error": f"Student {student_id} not found."}

    statuses = []
    for course_dict in student_data[student_id].get("status", []):
        if course in course_dict:
            statuses.extend(course_dict[course])

    return {"student_id": student_id, "course": course, "history": statuses}


# ----------------------
# Tool 2: Update status
# ----------------------
def update_student_course_status(student_id: str, course: str, current_session_feedback: str) -> Dict[str, Any]:
    """
    Use the LLM to convert session feedback into a structured status entry and update the DB.
    """
    # Fetch current record
    student_data = get_db_object().get_items_data(
        collection_name="students_db",
        item_ids=[student_id],
        id_col="student_id",
    )
    if not student_data or student_id not in student_data:
        return {"error": f"Student {student_id} not found."}

    llm = get_model()
    messages = [
        SystemMessage(content=UPDATE_STUDENT_STATUS_SYSTEM_PROMPT),
        HumanMessage(content=UPDATE_STUDENT_STATUS_USER_PROMPT.format(
            course=course,
            current_session_feedback=current_session_feedback
        )),
    ]
    resp = llm(messages)
    parsed = json_parser(getattr(resp, "content", str(resp)))
    parsed.update({"date": datetime.now().strftime("%d-%m-%Y")})

    # Append to the correct course
    found_course = False
    for course_dict in student_data[student_id].get("status", []):
        if course in course_dict:
            course_dict[course].append(parsed)
            found_course = True
            break

    if not found_course:
        student_data[student_id].setdefault("status", []).append({course: [parsed]})

    # Save
    get_db_object().update_metadata(
        collection_name="students_db",
        item_id=student_id,
        new_metadata=student_data[student_id]
    )

    return {"message": "Student status updated successfully.", "new_entry": parsed}

# -------------------------
# Tests for Student Tools
# -------------------------

def init_students_db_with_few_examples():
    STUDENTS_COLLECTION = "students_db"

    # Today's date in dd-mm-YYYY
    today_str = datetime.now().strftime("%d-%m-%Y")

    # Example students
    students_data = [
        {
            "student_id": "S001",
            "name": "Alice Johnson",
            "status": [
                {
                    "Math": [
                        {"score": 85, "note": "Good with algebra, needs work on geometry proofs", "date": today_str}
                    ]
                },
                {
                    "History": [
                        {"score": 92, "note": "Strong in American history, some gaps in world history", "date": today_str}
                    ]
                }
            ]
        },
        {
            "student_id": "S002",
            "name": "Bob Smith",
            "status": [
                {
                    "Science": [
                        {"score": 78, "note": "Understands physics basics, struggles with chemical equations", "date": today_str}
                    ]
                }
            ]
        },
        {
            "student_id": "S003",
            "name": "Charlie Davis",
            "status": [
                {
                    "SAT": [
                        {"score": 88, "note": "Reads passages quickly but needs inference question practice", "date": today_str}
                    ]
                },
                {
                    "Math": [
                        {"score": 80, "note": "Solid arithmetic, needs more practice in probability", "date": today_str}
                    ]
                }
            ]
        }
    ]

    # Convert to DataFrame
    df_students = pd.DataFrame(students_data)

    # Index into Qdrant (no embeddings needed, so need_to_embed_col=False)
    index_df(
        df=df_students,
        index_by_col="student_id",     # used to generate IDs
        need_to_embed_col=False,       # no semantic search needed here
        id_col="student_id",           # unique identifier
        collection_name=STUDENTS_COLLECTION
    )

    print(f"Inserted {len(df_students)} example students into '{STUDENTS_COLLECTION}' collection.")


def _ensure_student_exists(student_id: str, name: str, course: str) -> None:
    """
    Ensure a test student with a minimal schema exists in the students_db collection.
    If not found, create it with a single empty course history.
    """
    db = get_db_object()
    existing = db.get_items_data(
        collection_name=STUDENTS_COLLECTION,
        item_ids=[student_id],
        id_col="student_id",
    )

    if existing and student_id in existing:
        # Ensure the course key exists
        payload = existing[student_id]
        has_course = False
        for course_dict in payload.get("status", []):
            if course in course_dict:
                has_course = True
                break
        if not has_course:
            payload.setdefault("status", []).append({course: []})
            db.update_metadata(
                collection_name=STUDENTS_COLLECTION,
                item_id=student_id,
                new_metadata=payload,
            )
        return

    # Create new record using index_df (no embeddings needed)
    df = pd.DataFrame([{
        "student_id": student_id,
        "name": name,
        "status": [{course: []}],   # list of dicts; each course maps to a list of entries
    }])

    index_df(
        df=df,
        index_by_col="student_id",     # arbitrary when need_to_embed_col=False
        need_to_embed_col=False,       # store with dummy vectors (dim=1)
        id_col="student_id",
        collection_name=STUDENTS_COLLECTION,
    )


def _course_history_len(student_id: str, course: str) -> int:
    """Return how many status entries exist for (student, course)."""
    db = get_db_object()
    data = db.get_items_data(
        collection_name=STUDENTS_COLLECTION,
        item_ids=[student_id],
        id_col="student_id",
    )
    if not data or student_id not in data:
        return 0
    payload = data[student_id]
    for course_dict in payload.get("status", []):
        if course in course_dict:
            return len(course_dict[course])
    return 0


def _test_common_get_student_course_status():
    """
    Test Tool #1: GetStudentcourseStatus
    - Ensure a student exists
    - Fetch their status for a course
    - Validate structure and that it matches DB counts
    """
    student_id = "TEST_STUDENT_001"
    course = "Math"
    _ensure_student_exists(student_id, name="Testy McTestface", course=course)

    res = get_student_course_status(student_id=student_id, course=course)
    assert isinstance(res, dict) and res.get("student_id") == student_id and res.get("course") == course, \
        f"Unexpected structure from get_student_course_status: {res}"

    expected_len = _course_history_len(student_id, course)
    actual_len = len(res.get("history", []))
    assert actual_len == expected_len, \
        f"History length mismatch: expected {expected_len}, got {actual_len}"

    print("✅ _test_common_get_student_course_status passed.")


def _test_common_update_student_course_status():
    """
    Test Tool #2: UpdateStudentcourseStatus
    - Ensure a student exists
    - Measure course history length
    - Call update with synthetic session feedback
    - Verify history length increased and new entry contains score/note/date
    """
    student_id = "TEST_STUDENT_001"
    course = "Math"
    _ensure_student_exists(student_id, name="Testy McTestface", course=course)

    before_len = _course_history_len(student_id, course)

    # Synthetic feedback that the LLM will summarize to score+note
    feedback = (
        "Student solved basic addition quickly but hesitated on multi-step word problems. "
        "Made a place-value mistake once. Overall progress is good; needs practice with translating text to equations."
    )

    res = update_student_course_status(
        student_id=student_id,
        course=course,
        current_session_feedback=feedback,
    )
    assert isinstance(res, dict) and "new_entry" in res, f"Unexpected updater response: {res}"
    new_entry = res["new_entry"]
    assert "score" in new_entry and "note" in new_entry and "date" in new_entry, \
        f"Missing keys in new_entry: {new_entry}"

    after_len = _course_history_len(student_id, course)
    assert after_len == before_len + 1, \
        f"Expected course history to grow by 1 (before={before_len}, after={after_len})."

    # Light sanity checks
    assert isinstance(new_entry["score"], int) and 0 <= new_entry["score"] <= 100, \
        f"Score must be an int 0–100, got: {new_entry['score']}"
    assert isinstance(new_entry["note"], str) and len(new_entry["note"]) > 0, \
        f"Note must be a non-empty string, got: {new_entry['note']}"
    # Date format dd-mm-YYYY
    try:
        datetime.strptime(new_entry["date"], "%d-%m-%Y")
    except Exception as e:
        raise AssertionError(f"Date format invalid (expected dd-mm-YYYY), got: {new_entry['date']}") from e

    print("✅ _test_common_update_student_course_status passed.")


if __name__ == "__main__":
    # init_students_db_with_few_examples()
    _test_common_update_student_course_status()
    _test_common_get_student_course_status()

