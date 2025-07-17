import os
from pathlib import Path

repo_folder = Path.cwd()
print(repo_folder)

OPENAI_API_KEY = os.getenv("API_KEY")  # API key
CHAT_DEPLOYMENT_NAME = "team9-gpt4o"
EMBEDDING_DEPLOYMENT_NAME = "team9-embedding"

AZURE_OPENAI_ENDPOINT = "?"
API_VERSION = "?"

TOKEN_COUNT_FILE_PATH = repo_folder / 'tokens_count/total_tokens.csv'

print(f"Token count file path: {TOKEN_COUNT_FILE_PATH}")