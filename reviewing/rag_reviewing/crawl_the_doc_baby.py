import asyncio

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_tavily import TavilyCrawl, TavilyExtract, TavilyMap
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv(verbose=True)
tavily_crawl = TavilyCrawl()
tavily_extract = TavilyExtract()

tavily_map = TavilyMap()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)


async def main():
    print(" Hello motherfuckers it's tavily here")

    res = tavily_crawl.invoke({
        'url': 'https://docs.langchain.com/',
        'max_depth': 5,
        'extract_depth': 'advanced',
        'instructions': 'DeepAgents comprehensive explanation'
    })

    all_doc = [Document(page_content=result['raw_content'], metadata={'source': result['url']}) for result in
               res['results']]
    splitted_docs = text_splitter.split_documents(all_doc)

    t_map = tavily_map.invoke({
        'url': 'https://docs.langchain.com/',
        'max_depth': 5,
        'extract_depth': 'advanced',
        'instructions': 'DeepAgents comprehensive explanation'
    })

    print(f"tavily map result  {t_map}")

if __name__ == '__main__':
    asyncio.run(main())
