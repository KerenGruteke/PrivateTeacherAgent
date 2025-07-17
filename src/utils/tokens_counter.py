import csv
from src.utils.constants import TOKEN_COUNT_FILE_PATH


def token_count_to_csv(response):
    """
    Extracts token usage from an LLM response and appends it to a CSV file.

    The CSV file will store the number of prompt (input) and completion (output) tokens used.
    """
    token_usage = response.response_metadata["token_usage"]
    prompt_tokens = token_usage["prompt_tokens"]
    completion_tokens = token_usage["completion_tokens"]

    with open(TOKEN_COUNT_FILE_PATH, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([prompt_tokens, completion_tokens])


# Example usage with a mock response object
class MockResponse:
    def __init__(self, prompt_tokens, completion_tokens):
        self.response_metadata = {
            "token_usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens
            }
        }

# Create a fake response
fake_response = MockResponse(prompt_tokens=87, completion_tokens=143)

# Call the function
token_count_to_csv(fake_response)