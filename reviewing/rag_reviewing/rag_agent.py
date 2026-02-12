import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_pinecone import PineconeVectorStore
from langchain_tavily import TavilySearch
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

tavily_search = TavilySearch(
    max_results=5,
    include_raw_content=True,
    search_depth="advanced",
    country="Israel",
)


text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=0)
embeddings = OllamaEmbeddings(model='qwen3-embedding:latest')
vector_store = PineconeVectorStore(index_name=os.environ['INDEX_NAME'], embedding=embeddings)
retriever = vector_store.as_retriever()


@tool
def search_and_ingest(query: str) -> str:
    """Search the web using Tavily for the given query, ingest the results into
    Pinecone vector store, then retrieve the most relevant chunks and return them
    as context. Use this tool whenever you need to answer a question that requires
    up-to-date information from the web."""
    # 1. Search the web
    results = tavily_search.invoke({"query": query})

    # 2. Build documents from the results
    all_docs = []
    sources = []
    for result in results['results']:
        content = result.get('raw_content') or result.get('content', '')
        url = result.get('url', '')
        if content:
            all_docs.append(Document(
                page_content=content,
                metadata={"source": url},
            ))
            sources.append(url)

    if not all_docs:
        return "No results found for this query."

    # 3. Split and ingest into Pinecone
    chunks = text_splitter.split_documents(all_docs)
    vector_store.add_documents(chunks)

    # 4. Retrieve the most relevant chunks
    docs = retriever.invoke(query)
    context = "\n\n---\n\n".join(doc.page_content for doc in docs)

    return f"Sources: {', '.join(sources)}\n\nContext:\n{context}"


llm = ChatOllama(
    model='nemotron-3-nano:latest',
    temperature=0.1,
    validate_model_on_init=True,
)

agent = create_agent(
    model=llm,
    tools=[search_and_ingest],
    system_prompt=(
        "You are a research assistant with access to web search and a vector database. "
        "When the user asks a question, use the search_and_ingest tool to find relevant "
        "information from the web and store it in the knowledge base. "
        "Then answer the question based on the retrieved context. "
        "Always cite the source URLs in your answer."
    ),
)


def main():
    result = agent.invoke(
        {"messages": [
            {"role": "user", "content":"Explain to me what are TavilyCrawl,TavilyExtract, TavilyMap"}]}
    )

    for msg in result["messages"]:
        if msg.type == "ai" and msg.content and not msg.tool_calls:
            print(msg.content)
            break


if __name__ == '__main__':
    main()
