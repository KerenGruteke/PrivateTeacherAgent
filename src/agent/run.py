
from src.agent.main_private_teacher import init_private_teacher
from src.utils.user_response import get_student_response
from src.agent.prompts import WELCOME_PROMPT, USER_REQUEST_PROMPT
from src.agent.student_evaluator import  _ensure_student_exists
from src.utils.constants import VALID_COURSES

if __name__ == "__main__":
    name = get_student_response("Hi! What is your name?")
    course = None
    while course not in VALID_COURSES:
        course = get_student_response(WELCOME_PROMPT.format(name=name))
    
    student_id = _ensure_student_exists(name="Testy McTestface", course=course)
    user_message = get_student_response(USER_REQUEST_PROMPT.format(name=name, course=course))
    
    agent, prompt_text = init_private_teacher(student_id, course, user_message)
    agent.run(prompt_text)
