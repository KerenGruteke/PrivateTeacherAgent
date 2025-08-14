# apps/private_teacher_chat.py
import json
import os
import re
import sys
from typing import Any, Dict

import streamlit as st

# Make project importable when launching `streamlit run`
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# ---- Project functions (adjust imports if your paths differ) ----
from src.agent.question_RAG import generate_question_agent
from src.agent.answer_evaluator import evaluate_answer

# Optional (hidden) helpers; if missing, UI still works
try:
    from src.agent.coacher import get_coacher_response as coacher_call
except Exception:
    coacher_call = None

# If your generator already pulls student status internally, great.
# If not, you can import and pass notes explicitly.
# from src.agent.student_evaluator import get_student_topic_status


# -----------------------
# Utils
# -----------------------
def _extract_json(text: str) -> Dict[str, Any]:
    """Extract a JSON object from agent output."""
    if not text:
        return {}
    # direct JSON
    try:
        return json.loads(text)
    except Exception:
        pass
    # 'Final Answer:\n{...}'
    m = re.search(r'Final Answer:\s*(\{.*\})', text, flags=re.S)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    # first JSON object in text
    m2 = re.search(r'(\{.*\})', text, flags=re.S)
    if m2:
        try:
            return json.loads(m2.group(1))
        except Exception:
            pass
    return {"raw": text, "parse_warning": "Could not parse JSON cleanly."}


def _coacher_hidden_nudge(student_state: str) -> str:
    """Call coacher but keep the persona hidden. Returns a short nudge to blend into teacher's reply."""
    try:
        if coacher_call is None:
            return "Nice effort—try one small step next and focus on the key idea."
        msg = coacher_call(student_state)  # your version may print and return None
        if isinstance(msg, str) and msg.strip():
            return msg.strip()
        return "You’re making progress—keep going!"
    except Exception:
        return ""


# -----------------------
# Streamlit UI
# -----------------------
st.set_page_config(page_title="Private Teacher", page_icon="🎓", layout="centered")
st.title("🎓 Private Teacher")

with st.sidebar:
    st.header("Session")
    student_id = st.text_input("Student ID", value="S001")
    course = st.selectbox("Course", options=["Math", "Science", "History", "SAT"], index=0)
    st.caption("The chat shows only you and your teacher. Other tools work quietly in the background.")

# Session state
if "messages" not in st.session_state:
    st.session_state.messages = []  # list of dicts: {"role": "student"|"teacher", "content": str}
if "awaiting_answer" not in st.session_state:
    st.session_state.awaiting_answer = False
if "current_question" not in st.session_state:
    st.session_state.current_question = None  # parsed JSON of last question
if "raw_question" not in st.session_state:
    st.session_state.raw_question = ""

# Render history
for msg in st.session_state.messages:
    with st.chat_message("user" if msg["role"] == "student" else "assistant", avatar="🧑‍🎓" if msg["role"] == "student" else "👩‍🏫"):
        st.markdown(msg["content"])

# Chat input
user_text = st.chat_input("Type a request (e.g., 'Give me a question on algebra') or send your answer…")
if not user_text:
    st.stop()

# Append student message
st.session_state.messages.append({"role": "student", "content": user_text})
with st.chat_message("user", avatar="🧑‍🎓"):
    st.markdown(user_text)

# Decide intent: if awaiting an answer -> evaluate; else -> generate question
def teacher_says(text: str):
    st.session_state.messages.append({"role": "teacher", "content": text})
    with st.chat_message("assistant", avatar="👩‍🏫"):
        st.markdown(text)

if st.session_state.awaiting_answer and st.session_state.current_question:
    # Treat message as student's answer to the last question
    q = st.session_state.current_question
    solution = q.get("solution", "")
    question = q.get("question", "")
    course_used = q.get("course", course)

    eval_json_str = evaluate_answer(
        student_answer=user_text,
        solution=solution,
        question=question,
        topic=course_used,
    )
    eval_obj = _extract_json(eval_json_str)

    # Build teacher reply (blend in a short nudge, but keep 1 persona)
    correctness = eval_obj.get("correctness", "—")
    score = eval_obj.get("score", None)
    score_str = f"{score:.2f}" if isinstance(score, (int, float)) else "—"
    feedback = eval_obj.get("feedback", "")
    next_step = eval_obj.get("next_step", "")
    nudge = _coacher_hidden_nudge(
        f"Course: {course_used}. Correctness: {correctness}. Next: {next_step}."
    )

    reply_lines = [
        f"**Feedback:** {feedback}" if feedback else "Thanks for your answer!",
        f"**Correctness:** {correctness} · **Score:** {score_str}",
    ]
    if next_step:
        reply_lines.append(f"**Next step:** {next_step}")
    if nudge:
        reply_lines.append(nudge)

    teacher_says("\n\n".join(reply_lines))

    # Reset state after evaluation
    st.session_state.awaiting_answer = False
    st.session_state.current_question = None

else:
    # Treat message as a request for a new question
    raw = generate_question_agent(user_text, student_id=student_id)
    st.session_state.raw_question = raw
    q = _extract_json(raw)
    st.session_state.current_question = q if "question" in q else None

    # Teacher shows only the question (and optional hint)
    question_text = q.get("question", None)
    hint = q.get("hint", None)

    if question_text:
        msg = f"Here’s a new question for **{q.get('course', course)}**:\n\n{question_text}"
        if hint:
            msg += f"\n\n*Hint:* {hint}"
        teacher_says(msg)
        st.session_state.awaiting_answer = True
    else:
        # Fallback: show raw text if parsing failed
        teacher_says("Here’s a practice item for you:\n\n" + (raw if isinstance(raw, str) else json.dumps(q, ensure_ascii=False)))

