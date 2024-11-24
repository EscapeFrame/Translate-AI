from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
import os
load_dotenv()

if __name__ == "__main__":
    pdf_path = "/Users/pdh/Desktop/프로젝트/ Word-analogy/국어사전.pdf"
    loadder = PyPDFLoader(file_path=pdf_path)
    documents = loadder.load()
    spliter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=50, separator="\n")
    text = spliter.split_documents(documents)
    embedder = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
    PineconeVectorStore.from_documents(documents=text, embedding=embedder, index_name=os.getenv("INDEX_NAME"))
