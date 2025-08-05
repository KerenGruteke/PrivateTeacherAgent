from langchain.chat_models import AzureChatOpenAI
from pydantic import Field
from typing import Optional
from src.utils.tokens_counter import log_token_count_to_csv

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
