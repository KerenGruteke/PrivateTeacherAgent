import os

OPENAI_API_KEY = os.getenv("API_KEY")  # API key
CHAT_DEPLOYMENT_NAME = "team9-gpt4o"
EMBEDDING_DEPLOYMENT_NAME = "team9-embedding"

AZURE_OPENAI_ENDPOINT = "?"
API_VERSION = "?"

OUTPUT_FILE_PATH = 'tokens_count/total_tokens.txt'