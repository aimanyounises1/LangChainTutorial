import os
from typing import Dict, Any

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langchain_ollama import OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

llm = init_chat_model(
    "ollama:nemotron-3-nano:latest",
    temperature=0.1,
    profile={
        "tool_calling": True,
        "structured_output": True,
    },
)

embeddings = OllamaEmbeddings(model='qwen3-embedding:latest')
vectorstore = PineconeVectorStore(index_name=os.environ['INDEX_NAME'], embedding=embeddings)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)

retriever = vectorstore.as_retriever()


@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """
    Retrieves relevant context from the vector store based on the query.
    
    Args:
        query: The search query string.
        
    Returns:
        A tuple containing the serialized content and the original documents.
    """
    retrieve_doc = retriever.invoke(input=query, k=4)

    serialized_content = "\n\n".join(
        (f"Content source : {doc.metadata['source']} + \n Content :{doc.page_content}")
        for doc in retrieve_doc
    )
    return serialized_content, retrieve_doc


def run_llm(query: str) -> Dict[str, Any]:
    # Create the agent with retrieval tool
    system_prompt = (
        "You are a helpful AI assistant that answers questions about LangChain documentation. "
        "You have access to a tool that retrieves relevant documentation."
        "Use the tool to find relevant information before answering questions."
        "Always cite the sources you use in your answers, "
        "If you cannot find the answer in the retrieved documentation, say so."
    )

    agent = create_agent(system_prompt=system_prompt, model=llm, tools=[retrieve_context])
    messages = [{"role": "user", "content": query}]
    response = agent.invoke({"messages": messages})
    answer = response["messages"][-1].content

    context_doc = []

    for message in response['messages']:
        if isinstance(message, ToolMessage) and hasattr(message, "artifact"):

            if isinstance(message.artifact, list):
                context_doc.extend(message.artifact)

    return {
        "answer": answer,
        "context": context_doc
    }


if __name__ == '__main__':
    result = run_llm(query="What are the deepAgents?")
    print(result['answer'])
    print(result['context'])
