from functools import cached_property, lru_cache

from src.agent.prompts import INITIALIZE_HAND_IN_HAND_SYSTEM_PROMPT, INITIALIZE_HAND_IN_HAND_USER_PROMPT
from src.utils.constants import CHAT_DEPLOYMENT_NAME, AZURE_OPENAI_ENDPOINT, API_VERSION, DEBUG_MODE
from src.utils.LLM_utils import LoggingAzureChatOpenAI
from src.data.index_and_search import get_db_object
from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
from src.agent.answer_evaluator import evaluate_answer


@lru_cache(maxsize=1)
def get_model():
    llm = LoggingAzureChatOpenAI(
        agent_name="HAND_IN_HAND",
        azure_deployment=CHAT_DEPLOYMENT_NAME,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        openai_api_version=API_VERSION,
        openai_api_type="azure",
        temperature=0,
    )
    return llm


def get_student_answer(question: str):
    """
    Tool Name: Get Student Answer
    Description:
        Prompts the student to provide an answer to a given sub-question.
        This tool is intended to be used interactively during step-by-step
        guidance, where the AI tutor breaks the main question into smaller
        sub-questions and collects answers from the student.

    Args:
        question (str): The sub-question to ask the student.

    Returns:
        str: The student's answer as plain text.
    """
    return str(input(f"\nQuestion: {question}\nWrite your answer here: "))


def get_common_mistakes(course: str=None, question: str=None, **kwargs):
    """
    Tool Name: Get Common Mistakes
    Description:
        Searches the 'common_mistakes' database for errors frequently made in
        similar questions from the specified course. Intended to help the tutor
        identify where students typically struggle and provide targeted feedback.

    Args:
        course (str): The name of the course the question belongs to.
        question (str): The specific question or sub-question text.

    Returns:
        list[str]: A list of common mistakes relevant to the question.
    """
    qid = f"course: {course}\nQuestion: {question}"

    relevant_mistakes = get_db_object().search_by_query_vec(
        "common_mistakes",
        query=qid,
        top_k=3
    )

    common_mistakes = [
        mistake
        for q_mistakes in relevant_mistakes
        for mistake in q_mistakes["common_mistakes"]
    ]
    return common_mistakes


def present_message_to_user(message: str) -> str:
    """
    Tool Name: Present Message to User
    Description:
        Displays a message to the student.

    Args:
        message (str): The message to display to the student.

    Returns:
        Student reply (could be also empty and then you should continue lead the conversation)
    """
    print("\n\n🫱🫲🤖 AI Hand In Hand:") 
    print(str(message).replace("\\n", "\n"))
    return str(input("\n🎓 Student: "))

# 2. Define tools
tools = [
    Tool(
        name="present_message_to_user",
        func=present_message_to_user,
        description=("Present a message to the student and get their reply."),
    ),
    Tool(
        name="evaluate_answer",
        func=evaluate_answer,
        description=(
            "Evaluate a student's answer against the reference solution. "
            "Returns JSON with correctness, score (0–1), feedback, common_mistakes. "
            "Also logs/merges common mistakes into the 'common_mistakes' DB for future use."
        ),
    ),
    Tool(
        name="get_common_mistakes",
        func=get_common_mistakes,
        description=(
            "Searches the 'common_mistakes' database for errors frequently made "
            "in similar questions from the same course. Returns a list of "
            "common mistakes to aid in diagnosing student difficulties."
        )
    )
]

def hand_in_hand_agent(course: str = None, question: str = None, solution: str = None, student_answer: str = None, **kwargs):
    """
    Interactively guides a student through solving a question step-by-step and evaluates their answers.

    This function initializes a hand_in_hand_agent that:
    1. Breaks a question into smaller sub-steps.
    2. Guides the student interactively through each sub-step.
    3. Uses the evaluate_answer tool to assess the student's answers.
    4. Provides constructive feedback and suggestions for improvement.

    Args:
        course (str, optional): Name or identifier of the course.
        question (str, optional): The main question to solve.
        solution (str, optional): Reference solution for evaluation.
        student_answer (str, optional): The student's initial answer.
        **kwargs: Additional keyword arguments for extensibility.

    Returns:
        str: The agent's response including step-by-step guidance and evaluation.
    """
    llm = get_model()

    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=DEBUG_MODE,
    )

    prompt = INITIALIZE_HAND_IN_HAND_SYSTEM_PROMPT + "\n" + \
             INITIALIZE_HAND_IN_HAND_USER_PROMPT.format(
                 course=course,
                 question=question,
                 reference_solution=solution,
                 student_answer=student_answer
             )

    response = agent.run(prompt)
    return response



if __name__ == "__main__":
    course = "Math"
    question = "Solve for x: 2x + 3 = 7"
    solution = "x = 2"
    student_answer = "x = 5"

    result = hand_in_hand_agent(course, question, solution, student_answer)
    print("\nFinal Output:\n", result)