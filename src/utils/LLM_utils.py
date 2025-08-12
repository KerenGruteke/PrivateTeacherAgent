from langchain.chat_models import AzureChatOpenAI
from openai import AzureOpenAI
from typing import Optional, List
from pydantic import Field
from tqdm import tqdm
from langchain.schema import SystemMessage, HumanMessage, AIMessage, BaseMessage
from src.utils.constants import EMBEDDING_DEPLOYMENT_NAME, AZURE_OPENAI_ENDPOINT, API_VERSION, EMBEDDING_MODEL
from src.utils.tokens_counter import log_token_count_to_csv
from dotenv import load_dotenv
import os

load_dotenv()

class LoggingAzureChatOpenAI(AzureChatOpenAI):
    agent_name: Optional[str] = Field(default="default_agent")

    def generate(self, messages: list[list[BaseMessage]], **kwargs):
        """
        messages: A list of lists (batch), each containing LangChain message objects
        """
        # Call the original generate
        response = super().generate(messages, **kwargs)

        # Go over each batch item
        for idx, message_list in enumerate(messages):
            # Extract system and user messages
            system_prompts = [m.content for m in message_list if isinstance(m, SystemMessage)]
            user_prompts = [m.content for m in message_list if isinstance(m, HumanMessage)]

            # Join if multiple system or user prompts exist
            system_prompt_text = "\n".join(system_prompts)
            user_prompt_text = "\n".join(user_prompts)

            # Token usage (LangChain's ChatResult)
            token_usage = response.llm_output.get("token_usage", {})
            prompt_tokens = token_usage.get("prompt_tokens", None)
            completion_tokens = token_usage.get("completion_tokens", None)

            # Generated answer
            generated_answer = str(response.generations[idx][0].text)

            union_prompt = f"System: {system_prompt_text}\nUser: {user_prompt_text}"

            # Log
            log_token_count_to_csv(
                self.agent_name,
                union_prompt,
                generated_answer,
                prompt_tokens,
                completion_tokens
            )

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
        for text in tqdm(texts, desc="Generating embeddings"):
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
    