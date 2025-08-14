

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
    print("\n\n📚🤖 AI Teacher:") 
    print(str(message).replace("\\n", "\n"))
    return str(input("\n🎓 Student: "))

def get_student_response(message):
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
    print("\n\n📚🤖 AI Teacher:\t")
    print(str(message).replace("\\n", "\n"))
    return str(input("\n🎓 Student: "))

if __name__ == "__main__":
    message_with_tabs = "check\\n\\ncheck2"
    present_message_to_user(message_with_tabs)