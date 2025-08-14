# apps/private_teacher_chat.py
import os
import re
import sys
from typing import List

import streamlit as st

# Make project importable when launching `streamlit run`
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.agent.main_private_teacher import init_private_teacher  # uses your tools under the hood

# ---------- Small helper ----------
def _final_answer(text: str) -> str:
    """Extract the teacher's final message from ReAct-style output."""
    if not isinstance(text, str):
        return str(text)
    m = re.search(r"Final Answer:\s*(.*)\Z", text, flags=re.S)
    return m.group(1).strip() if m else text.strip()

# ---------- Page setup ----------
st.set_page_config(page_title="Private Teacher", page_icon="🎓", layout="centered")
st.title("🎓 Private Teacher")

with st.sidebar:
    st.header("Session")
    student_id = st.text_input("Student ID", value="S001")
    course = st.selectbox("Course", ["Math", "Science", "History", "SAT"], index=0)
    st.caption("Chat with your teacher. Other helpers run quietly in the background.")

# ---------- Session state ----------
if "messages" not in st.session_state:
    # Each message: {"role": "student" | "teacher", "content": str}
    st.session_state.messages: List[dict] = []
if "warm_welcome" not in st.session_state:
    welcome = (
        f"Hi! I’m your private teacher for **{course}**. "
        "Tell me what you’d like to practice, or say “give me a question”."
    )
    st.session_state.messages.append({"role": "teacher", "content": welcome})
    st.session_state.warm_welcome = True

# ---------- Render history ----------
for msg in st.session_state.messages:
    role = "user" if msg["role"] == "student" else "assistant"
    avatar = "🧑‍🎓" if msg["role"] == "student" else "👩‍🏫"
    with st.chat_message(role, avatar=avatar):
        st.markdown(msg["content"])

# ---------- Chat input ----------
user_msg = st.chat_input("Type a request (e.g., 'Give me a question on algebra')…")
if not user_msg:
    st.stop()

# Show student bubble immediately
st.session_state.messages.append({"role": "student", "content": user_msg})
with st.chat_message("user", avatar="🧑‍🎓"):
    st.markdown(user_msg)

# ---------- Call Main Private Teacher agent ----------
# Build a fresh agent each turn (simple & stateless). Your init function pulls student notes and tools.
agent, prompt_text = init_private_teacher(student_id=student_id, course=course, user_message=user_msg)

# Run the agent. (AgentType.ZERO_SHOT_REACT_DESCRIPTION expects 'Final Answer:' at the end.)
raw_reply = agent.run(prompt_text)  # if you migrate to .invoke later, swap this line accordingly
teacher_reply = _final_answer(raw_reply)

# Show the teacher bubble
st.session_state.messages.append({"role": "teacher", "content": teacher_reply})
with st.chat_message("assistant", avatar="👩‍🏫"):
    st.markdown(teacher_reply)

# ---------- Footer ----------
st.markdown("---")
st.caption("Tip: Ask for a question, answer it, and I’ll give feedback and next steps.")
