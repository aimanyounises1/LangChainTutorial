import asyncio
import os
from typing import List, Any, Dict

from dotenv import load_dotenv
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

# MUST load env vars BEFORE importing langchain_tavily
load_dotenv(verbose=True)

from langchain_tavily import TavilyMap, TavilyCrawl, TavilySearch, TavilyExtract

# Initialize embeddings model using Ollama
embeddings = OllamaEmbeddings(model='qwen3-embedding:latest')

# Initialize Pinecone vector store with the specified index and embeddings
vectorstore = PineconeVectorStore(index_name=os.environ['INDEX_NAME'], embedding=embeddings)

# Initialize Tavily tools for crawling, searching, mapping, and extraction
tavily_crawl = TavilyCrawl()
tavily_search = TavilySearch()
tavily_map = TavilyMap()
tavily_extract = TavilyExtract()


def chunk_urls(urls: List[str], chunk_size: int = 20) -> List[List[str]]:
    """
    Splits a list of URLs into smaller chunks.

    Args:
        urls: The list of URLs to split.
        chunk_size: The size of each chunk.

    Returns:
        A list of lists, where each inner list is a chunk of URLs.
    """
    chunks = []
    for i in range(0, len(urls), chunk_size):
        chunk = urls[i:i + chunk_size]
        chunks.append(chunk)
    return chunks


async def extract_batch(urls: List[str], batch_num: int) -> List[Dict[str, Any]]:
    """
    Asynchronously extracts content from a batch of URLs using Tavily Extract.

    Args:
        urls: A list of URLs to extract content from.
        batch_num: The batch number for logging purposes.

    Returns:
        The extraction results from Tavily.
    """
    try:
        docs = await tavily_extract.ainvoke({"urls": urls})
        print(f"Batch {batch_num} extracted {len(docs.get('results', []))} documents")
        return docs
    except Exception as e:
        print(e)
        return []


async def async_extract(url_batches: List[List[str]]):
    """
    Orchestrates the asynchronous extraction of content from multiple batches of URLs.

    Args:
        url_batches: A list of URL batches.

    Returns:
        A list of Document objects containing the extracted content.
    """
    # Create a list of coroutines for extracting each batch
    tasks = [extract_batch(batch, i + 1) for i, batch in enumerate(url_batches)]  # tasks is corutonies object
    
    # Run all extraction tasks concurrently
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    all_pages = []
    failed_batches = 0
    
    # Process results from all batches
    for result in results:
        if isinstance(result, Exception):
            failed_batches += 1
            print(f"Failed batch {failed_batches}")
        else:
            # Convert extraction results to Document objects
            for extracted_batch in result['results']:
                document = Document(page_content=extracted_batch['raw_content'],
                                    metadata={'source': extracted_batch['url']})
                all_pages.append(document)
    return all_pages


async def index_documents_async(documents: List[Document], batch_size: int):
    """
    Asynchronously indexes documents into the vector store in batches.

    Args:
        documents: The list of documents to index.
        batch_size: The number of documents per batch.

    Returns:
        The number of successfully indexed batches.
    """
    # Create batches of documents
    batches = [
        documents[i: i + batch_size] for i in range(0, len(documents), batch_size)
    ]

    async def add_batch(batch: List[Document], batch_num: int):
        """Helper function to add a single batch to the vector store."""
        try:
            await vectorstore.aadd_documents(batch)
            print(f"Batch {batch_num} added to {len(batch)} documents")
        except Exception as e:
            print(f"Batch {batch_num} failed to add to {len(batch)} documents")
            print(f"Failed batch {batch_num} error is {e}")
            return False
        return True

    # Create tasks for adding each batch
    tasks = [add_batch(batch, i + 1) for i, batch in enumerate(batches)]
    
    # Run all indexing tasks concurrently
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    successful_batches = sum(1 for result in results if result is True)
    if successful_batches == len(batches):
        print(f"All batches added to {len(batches)} documents")
    else:
        print(f"Failed batches added to {len(batches)} documents")
    return successful_batches


async def main():
    # Example of how to initialize LLM (currently unused in this script)
    # llm = ChatOllama(model='qwen3:30b-a3b',
    #                  validate_model_on_init=True,
    #                  temperature=0.8,
    #                  reasoning=True
    #                  )

    # Example: Crawl a specific URL (results are printed but not used for indexing below)
    res = tavily_crawl.invoke({
        "url": "https://docs.langchain.com/oss/python/langchain/overview",
        "max_depth": 3,
        "extract_depth": "advanced"
    })
    # print(res['results'])
    for result in res['results']:
        print(result)
    
    # Convert crawl results to Documents (Note: This variable 'all_docs' is overwritten later)
    all_docs = [Document(page_content=str(result['raw_content']),
                         metadata={"source": result['url']})
                for result in res['results']]

    # 1. Map the website to get a list of URLs
    site_map = tavily_map.invoke({
        "url": "https://python.langchain.com/",
        "max_depth": 3,
        "extract_depth": "advanced"
    })

    # 2. Chunk the URLs for batch processing
    url_batches = chunk_urls(site_map['results'], chunk_size=5)
    
    # 3. Extract content from the URLs asynchronously
    all_docs = await async_extract(url_batches)
    
    # 4. Split the extracted text into smaller chunks for embedding
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
    splitted_docs = text_splitter.split_documents(all_docs)
    
    # 5. Index the split documents into the vector store
    await index_documents_async(splitted_docs, batch_size=500)

    print("Documentation Pipeline ingestion completed successfully")
    print("================")
    print(all_docs)
    print('\n')
    print(url_batches)
    print('\n')
    print("======================")
    print(site_map)
    print('\n')
    print("======================")


if __name__ == '__main__':
    asyncio.run(main())
