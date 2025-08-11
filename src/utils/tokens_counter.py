import csv
from src.utils.folders_utils import get_token_count_file_path
import time


def log_token_count_to_csv(agent_name, prompt, generated_answer, prompt_tokens, completion_tokens):
    """
    Extracts token usage from an LLM response and appends it to a CSV file.

    The CSV file will store the number of prompt (input) and completion (output) tokens used.
    """
    date_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    with open(get_token_count_file_path(), mode="a", newline="", encoding='utf-8-sig') as file:
        writer = csv.writer(file, lineterminator="\n")
        # dont save the whole prompt, but only the first 1000 characters
        prompt = prompt[:1000]
        # dont save the whole generated answer, but only the first 1000 characters
        generated_answer = generated_answer[:1000]
    
        writer.writerow([date_time, agent_name, prompt, generated_answer, prompt_tokens, completion_tokens])
