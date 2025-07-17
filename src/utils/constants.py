import os
from pathlib import Path


def get_repo_folder():
    """Find the repository folder by traversing up the directory tree."""
    current = Path.cwd()
    for parent in current.parents:
        if parent.name == "PrivateTeacherAgent":
            return parent
    raise FileNotFoundError("Repository folder 'PrivateTeacherAgent' not found.")


OPENAI_API_KEY = os.getenv("API_KEY")  # API key
CHAT_DEPLOYMENT_NAME = "team9-gpt4o"
EMBEDDING_DEPLOYMENT_NAME = "team9-embedding"

AZURE_OPENAI_ENDPOINT = "https://096290-oai.openai.azure.com"
API_VERSION = "2023-05-15"

TOKEN_COUNT_FILE_PATH = get_repo_folder() / os.path.join('tokens_count','total_tokens.csv')