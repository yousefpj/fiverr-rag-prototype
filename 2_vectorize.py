import os
import time
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load API keys from .env
load_dotenv()

INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "rag-prototype")

def vectorize_and_upload(file_path):
    # 1. Load and chunk the PDF
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    print(f"Prepared {len(chunks)} chunks from {file_path}.")

    # 2. Initialize Pinecone and create the index if it doesn't exist
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    
    if not pc.has_index(INDEX_NAME):
        print(f"Creating Pinecone index '{INDEX_NAME}'...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=1536,  # Matches OpenAI text-embedding-3-small
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        time.sleep(5)  # Give Pinecone a few seconds to initialize

    # 3. Embed and upsert chunks to Pinecone
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    print("Uploading vectors to Pinecone...")
    PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=INDEX_NAME
    )
    print(f"Success! {len(chunks)} chunks are now live in Pinecone index '{INDEX_NAME}'.")

if __name__ == "__main__":
    vectorize_and_upload("sample.pdf")