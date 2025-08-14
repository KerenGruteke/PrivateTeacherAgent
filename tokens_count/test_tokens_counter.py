from src.utils.tokens_counter import log_token_count_to_csv

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
fake_response = MockResponse(prompt_tokens=55, completion_tokens=20)
log_token_count_to_csv("agent_1", fake_response)

fake_response = MockResponse(prompt_tokens=87, completion_tokens=143)
log_token_count_to_csv("agent_1", fake_response)

fake_response = MockResponse(prompt_tokens=122, completion_tokens=20)
log_token_count_to_csv("agent_2", fake_response)