from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_and_chunk_pdf(file_path):
    # 1. Load the PDF
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    print(f"Loaded {len(documents)} page(s) from {file_path}.")

    # 2. Split into chunks (1000 chars with 200 char overlap to preserve context)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks.")
    
    return chunks

if __name__ == "__main__":
    chunks = load_and_chunk_pdf("sample.pdf")
    
    # Preview the first chunk and its metadata (used for source citations)
    print("\n--- SAMPLE CHUNK 1 ---")
    print(chunks[0].page_content)
    print("\n--- METADATA ---")
    print(chunks[0].metadata)