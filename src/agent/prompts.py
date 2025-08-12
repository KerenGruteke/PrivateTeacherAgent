# --------------------
# Main Teacher Prompts
# --------------------

welcome_prompt = """
Hi there! 👋 I’m your private teacher for today—ready to help you learn, practice, and improve.
My main areas of expertise are Math, History, Science, and SAT questions, but you’re welcome to ask me about other topics too.

What topic would you like to focus on today?
"""

# --------------------
# Question RAG Prompts
# --------------------
GET_QUERY_TO_SEARCH_SYSTEM_PROMPT = """
{query_to_search}
"""
GET_QUERY_TO_SEARCH_USER_PROMPT = """
"""
REWRITE_QUESTION_SYSTEM_PROMPT = """
"""
REWRITE_QUESTION_USER_PROMPT = """
"""


# --------------------------
# Student Evaluator Prompts
# --------------------------
UPDATE_STUDENT_STATUS_SYSTEM_PROMPT = """
"""
UPDATE_STUDENT_STATUS_USER_PROMPT = """
"""


# ---------------
# Coacher Prompts
# ---------------
COACHER_SYSTEM_PROMPT = """
"""
COACHER_USER_PROMPT = """
"""


# -----------------------
# Evaluate Answer Prompts
# -----------------------
EVALUATE_ANSWER_SYSTEM_PROMPT = """
"""
EVALUATE_ANSWER_USER_PROMPT = """
"""

# -------------------------------
# Initialize hand in hand Prompts
# -------------------------------
INITIALIZE_HAND_IN_HAND_SYSTEM_PROMPT = """
"""
INITIALIZE_HAND_IN_HAND_USER_PROMPT = """
"""


# ---------------------
# Final feedback Prompt
# ---------------------
FINAL_FEEDBACK_SYSTEM_PROMPT = """
"""
FINAL_FEEDBACK_USER_PROMPT = """
"""