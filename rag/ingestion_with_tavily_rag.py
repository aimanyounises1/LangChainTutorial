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

embeddings = OllamaEmbeddings(model='qwen3-embedding:latest')
vectorstore = PineconeVectorStore(index_name=os.environ['INDEX_NAME'], embedding=embeddings)
tavily_crawl = TavilyCrawl()
tavily_search = TavilySearch()
tavily_map = TavilyMap()
tavily_extract = TavilyExtract()


def chunk_urls(urls: List[str], chunk_size: int = 20) -> List[List[str]]:
    chunks = []
    for i in range(0, len(urls), chunk_size):
        chunk = urls[i:i + chunk_size]
        chunks.append(chunk)
    return chunks


async def extract_batch(urls: List[str], batch_num: int) -> List[Dict[str, Any]]:
    try:
        docs = await tavily_extract.ainvoke({"urls": urls})
        print(f"Batch {batch_num} extracted {len(docs.get('results', []))} documents")
        return docs
    except Exception as e:
        print(e)
        return []


async def async_extract(url_batches: List[List[str]]):
    tasks = [extract_batch(batch, i + 1) for i, batch in enumerate(url_batches)]  # tasks is corutonies object
    results = await asyncio.gather(*tasks, return_exceptions=True)
    all_pages = []
    failed_batches = 0
    for result in results:
        if isinstance(result, Exception):
            failed_batches += 1
            print(f"Failed batch {failed_batches}")
        else:
            for extracted_batch in result['results']:
                document = Document(page_content=extracted_batch['raw_content'],
                                    metadata={'source': extracted_batch['url']})
                all_pages.append(document)
    return all_pages


async def index_documents_async(documents: List[Document], batch_size: int):
    # Create batches
    batches = [
        documents[i: i + batch_size] for i in range(0, len(documents), batch_size)
    ]

    async def add_batch(batch: List[Document], batch_num: int):
        try:
            await vectorstore.aadd_documents(batch)
            print(f"Batch {batch_num} added to {len(batch)} documents")
        except Exception as e:
            print(f"Batch {batch_num} failed to add to {len(batch)} documents")
            print(f"Failed batch {batch_num} error is {e}")
            return False
        return True

    tasks = [add_batch(batch, i + 1) for i, batch in enumerate(batches)]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    successful_batches = sum(1 for result in results if result is True)
    if successful_batches == len(batches):
        print(f"All batches added to {len(batches)} documents")
    else:
        print(f"Failed batches added to {len(batches)} documents")
    return successful_batches


async def main():
    # llm = ChatOllama(model='qwen3:30b-a3b',
    #                  validate_model_on_init=True,
    #                  temperature=0.8,
    #                  reasoning=True
    #                  )
    res = tavily_crawl.invoke({
        "url": "https://docs.langchain.com/oss/python/langchain/overview",
        "max_depth": 3,
        "extract_depth": "advanced"
    })
    # print(res['results'])
    for result in res['results']:
        print(result)
    all_docs = [Document(page_content=str(result['raw_content']),
                         metadata={"source": result['url']})
                for result in res['results']]

    site_map = tavily_map.invoke({
        "url": "https://python.langchain.com/",
        "max_depth": 3,
        "extract_depth": "advanced"
    })

    url_batches = chunk_urls(site_map['results'], chunk_size=5)
    all_docs = await async_extract(url_batches)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
    splitted_docs = text_splitter.split_documents(all_docs)
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
