from src.utils.LLM_utils import LoggingEmbedding


embeder_client = LoggingEmbedding()

embeddings = embeder_client.embed(["a"])

1 == 1