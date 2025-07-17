import csv
from src.utils.constants import TOKEN_COUNT_FILE_PATH
import time


def log_token_count_to_csv(agent_name, prompt, generated_answer, prompt_tokens, completion_tokens):
    """
    Extracts token usage from an LLM response and appends it to a CSV file.

    The CSV file will store the number of prompt (input) and completion (output) tokens used.
    """
    date_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    with open(TOKEN_COUNT_FILE_PATH, mode="a", newline="", encoding='utf-8-sig') as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow([date_time, agent_name, prompt, generated_answer, prompt_tokens, completion_tokens])
