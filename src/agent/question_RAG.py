from src.agent.prompts import get_query_to_search_prompt
from src.utils.LLM_utils import SystemMessage, HumanMessage

def get_question_using_RAG(RAG_agent, user_request):
    query_to_search, collection_name = get_query_to_search(RAG_agent, user_request)
    retrieved_docs = search_documents(query_to_search, collection_name)
    rewritten_question, answer_and_metadata = rewrite_question(query_to_search, retrieved_docs, collection_name)
    return rewritten_question, answer_and_metadata

def get_query_to_search(RAG_agent, user_request):
    # call the llm API using get_query_to_search_prompt
    response = RAG_agent.generate([
        [
            SystemMessage(content="You are a helpful assistant that answers politely."),
            HumanMessage(content=user_request)
        ]
    ])
    
def get_question_RAG_agent():
    llm = LoggingAzureChatOpenAI(
        agent_name="question_RAG",
        azure_deployment=CHAT_DEPLOYMENT_NAME,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        openai_api_version=API_VERSION,
        openai_api_type="azure",
        temperature=0,
    )
    return llm