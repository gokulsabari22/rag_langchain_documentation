import os
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Pinecone
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
import pinecone
from const import INDEX_NAME
from dotenv import load_dotenv

load_dotenv()
pinecone.init(api_key=os.environ["PINECONE_API_KEY"], environment=os.environ["PINECONE_ENVIRONMENT_REGION"])

def run_llm(query, chat_history):
    embedding = OpenAIEmbeddings()
    llm = ChatOpenAI(verbose=True, model="gpt-3.5-turbo", temperature=0.0)
    doc_search = Pinecone.from_existing_index(INDEX_NAME, embedding)
    docs = doc_search.similarity_search(query)
    qa = ConversationalRetrievalChain.from_llm(llm=llm, retriever=doc_search.as_retriever())
    result = qa({"question": query, "chat_history": chat_history})
    return result["answer"]



if __name__ == "__main__":
    query = "When the elections will be held in India in 2024?"
    result = run_llm(query=query)
    print(result)