import os
from langchain.document_loaders import ReadTheDocsLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
import pinecone
from const import INDEX_NAME
from dotenv import load_dotenv
import time

load_dotenv()
pinecone.init(api_key=os.environ["PINECONE_API_KEY"], environment=os.environ["PINECONE_ENVIRONMENT_REGION"])

def ingest_docs():
    loader = ReadTheDocsLoader(path="langchain-docs/api.python.langchain.com/en/latest", encoding="utf-8")
    raw_document = loader.load()
    print(f"Loaded {len(raw_document)} documents")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=0)
    documents = text_splitter.split_documents(raw_document)
    print(f"Splitted into {len(documents)} chunks")

    for doc in documents:
        old_path = doc.metadata["source"]
        new_url = old_path.replace("langchain-docs", "https:/")
        doc.metadata.update({"source": new_url})

    embeddings = OpenAIEmbeddings()
    for i in range(5):
        Pinecone.from_documents(documents, embeddings, index_name=INDEX_NAME)
        if i==49:
            time.sleep(100) 
    print("Added to Pinecone Vector DB")

if __name__ == "__main__":
    ingest_docs()