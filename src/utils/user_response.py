def present_question(question: str) -> str:
    """
    Tool Name: Get Student Answer
    Description:
        Prompts the student to provide an answer to a given question.
        This tool is intended to be used interactively during student interaction
        where the AI tutor persent the question to the student collects answers from the student.

    Args:
        question (str): The question to ask the student.

    Returns:
        str: The student's answer as plain text.
    """
    return str(input(f"\nQuestion: {question}\nWrite your answer here: "))

def get_student_response(message_to_student):
    """
    Tool Name: Get Student Response
    Description:
        Prompts the student to provide a response to a given message.
        This tool is intended to be used interactively during student interaction
        where the AI tutor presents the message to the student and collects their response.

    Args:
        message_to_student (str): The message to present to the student.

    Returns:
        str: The student's response as plain text.
    """
    return str(input(f"\n{message_to_student}\nYour Reply: "))