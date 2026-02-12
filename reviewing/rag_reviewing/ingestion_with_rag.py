import asyncio
import os

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_pinecone import PineconeVectorStore
from langchain_tavily import TavilySearch
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()
tavily_tool = TavilySearch(max_results=5, include_image_descriptions=True, country="Israel", search_depth="advanced",
                           include_raw_content=True)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=0)
embeddings = OllamaEmbeddings(model='qwen3-embedding:latest')
vector_store = PineconeVectorStore(index_name=os.environ['INDEX_NAME'], embedding=embeddings)
retriever = vector_store.as_retriever()
prompt_template = ChatPromptTemplate.from_template("""
    Answer the questions based on the below context
    {context}

    Question: {question}

    Provide a detailed answer
""")
llm = ChatOllama(model='nemotron-3-nano:latest',
                 temperature=0.1,
                 reasoning=True,
                 validate_model_on_init=True
                 )


async def ingest_docs(query: str):
    """Search with Tavily, split results into chunks, and ingest into Pinecone."""
    results = tavily_tool.invoke({"query": query})
    all_docs = []
    for result in results['results']:
        content = result.get('raw_content') or result.get('content', '')
        if content:
            all_docs.append(Document(
                page_content=content,
                metadata={"source": result.get('url', '')}
            ))
    splitted_docs = text_splitter.split_documents(all_docs)
    await vector_store.aadd_documents(splitted_docs)
    print(f"Ingested {len(splitted_docs)} chunks from {len(all_docs)} documents")
    return splitted_docs


async def main():
    query = "What is the Pinecone Database? For what purposes it is being used?"

    # 1. Ingest docs from Tavily search into Pinecone
    await ingest_docs(query)

    # 2. Retrieve relevant documents from the vector store
    docs = await retriever.ainvoke(query)
    context = "\n\n".join(doc.page_content for doc in docs)

    # 3. Answer the question using the retrieved context
    chain = prompt_template | llm | StrOutputParser()
    response = await chain.ainvoke({"context": context, "question": query})
    print(response)


if __name__ == '__main__':
    asyncio.run(main())
