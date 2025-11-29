import os

from dotenv import load_dotenv
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv(verbose=True)

if __name__ == '__main__':
    llm = ChatOllama(model='qwen3:30b-a3b',
                     validate_model_on_init=True,
                     temperature=0.8,
                     reasoning=True
                     )
    embeddings = OllamaEmbeddings(model='qwen3-embedding:latest')
    query = 'List associated skills the applicants demonstrate in Python'
    chain = PromptTemplate.from_template(
        template=query ) | llm # Pass the prompt to llm directly using LCEL


    vectorstore = PineconeVectorStore(index_name=os.environ['INDEX_NAME'], embedding=embeddings)
    retrieval_qa_chat_prompt =  hub.pull('langchain-ai/retrieval-qa-chat')
    combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
    retrieval_chain = create_retrieval_chain(retriever=vectorstore.as_retriever(), combine_docs_chain=combine_docs_chain)
    result = retrieval_chain.invoke({"input": query})