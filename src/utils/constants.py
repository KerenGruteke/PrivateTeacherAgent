import os

# LLM 
OPENAI_API_KEY = os.getenv("API_KEY")  # API key
CHAT_DEPLOYMENT_NAME = "team9-gpt4o"
EMBEDDING_DEPLOYMENT_NAME = "team9-embedding"

AZURE_OPENAI_ENDPOINT = "https://096290-oai.openai.azure.com"
API_VERSION = "2023-05-15"

PRICE_1M_INPUT_TOKENS = 2.5
PRICE_1M_OUTPUT_TOKENS = 10

# QDRANT DB
QDRANT_CLUSTER_URL = "https://23beef8e-a598-4086-8a43-360a6973c7e3.us-west-2-0.aws.cloud.qdrant.io:6333"
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")