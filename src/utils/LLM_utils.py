from langchain.chat_models import AzureChatOpenAI
from openai import AzureOpenAI
from typing import Optional, List
from pydantic import Field

from src.utils.constants import EMBEDDING_DEPLOYMENT_NAME, AZURE_OPENAI_ENDPOINT, API_VERSION, EMBEDDING_MODEL
from src.utils.tokens_counter import log_token_count_to_csv
from dotenv import load_dotenv
import os

load_dotenv()

class LoggingAzureChatOpenAI(AzureChatOpenAI):
    agent_name: Optional[str] = Field(default="default_agent")

    def generate(self, messages, **kwargs):
        response = super().generate(messages, **kwargs)

        prompt = "\n".join(msg.content for msg in messages[0])
        token_usage = response.llm_output["token_usage"]
        prompt_tokens = token_usage["prompt_tokens"]
        completion_tokens = token_usage["completion_tokens"]
        generated_answer = str(response.generations[0][0].text)

        log_token_count_to_csv(self.agent_name, prompt, generated_answer, prompt_tokens, completion_tokens)

        return response


class LoggingEmbedding:

    def __init__(self):
        self.client = AzureOpenAI(
            azure_deployment=EMBEDDING_DEPLOYMENT_NAME,
            api_version=API_VERSION,
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_key=os.getenv("OPENAI_API_KEY"),
        )

    def embed(self, texts: List[str]):
        embeddings = []
        for text in texts:
            resp = self.client.embeddings.create(input=text, model=EMBEDDING_MODEL)
            embedding_vec = resp.data[0].embedding
            embeddings.append(embedding_vec)

            input_tokens = resp.usage.prompt_tokens
            log_token_count_to_csv("Embedded", text, "Vector", input_tokens, 0)

        dim = len(embeddings[0]) if embeddings else 0
        return embeddings, dim
    
if __name__ == "__main__":
    # Example usage
    embeder_client = LoggingEmbedding()
    embeddings, dim = embeder_client.embed(["a"])
    print(f"Embedding dimension: {dim}")
    1 == 1
    
    # simple df 
    import pandas as pd
    df = pd.DataFrame({
        "id": [1, 2, 3],
        "text": ["Hello", "World", "Test"]
    })
    
    # Convert to embeddings
    df['embeddings'], dim = embeder_client.embed(df['text'].tolist())
    print(f"Embedding dimension: {dim}")
    print(df)