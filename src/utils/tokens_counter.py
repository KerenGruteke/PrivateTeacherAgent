import csv
from src.utils.constants import TOKEN_COUNT_FILE_PATH
import time

def log_token_count_to_csv(agent_name, response):
    """
    Extracts token usage from an LLM response and appends it to a CSV file.

    The CSV file will store the number of prompt (input) and completion (output) tokens used.
    """
    token_usage = response.response_metadata["token_usage"]
    prompt_tokens = token_usage["prompt_tokens"]
    completion_tokens = token_usage["completion_tokens"]
    date_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    with open(TOKEN_COUNT_FILE_PATH, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([date_time, agent_name, prompt_tokens, completion_tokens])
