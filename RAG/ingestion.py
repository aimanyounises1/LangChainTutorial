import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_ollama import OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import CharacterTextSplitter

load_dotenv(verbose=True)

if __name__ == '__main__':
    print("Ingesting ...")
    loader =  TextLoader('/Users/aimanyounis/Downloads/NVIDIA-CV-Enhancement-Guide.txt')
    document = loader.load()
    print(document)

    # Text Splitter

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(document)
    print(len(texts))
    embeddings = OllamaEmbeddings(model='qwen3-embedding:latest')
    PineconeVectorStore.from_documents(texts,embeddings, index_name=os.environ['INDEX_NAME'])
    