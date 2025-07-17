import csv
from src.utils.constants import OUTPUT_FILE_PATH


# TODO: change the function
def write_token_count_to_csv(response):
    """gets an llm response and store the number of tokens used into a csv file"""
    token_usage = response.response_metadata["token_usage"]
    input_token_counter, output_token_counter = token_usage["prompt_tokens"], token_usage["completion_tokens"]

    with open(OUTPUT_FILE_PATH, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([input_token_counter, output_token_counter])