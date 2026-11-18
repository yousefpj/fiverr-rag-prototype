import os
import tempfile
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load environment variables
load_dotenv()
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "rag-prototype")
APP_PASSWORD = os.getenv("APP_PASSWORD", "changeme123")

# Page setup
st.set_page_config(page_title="AI Document Assistant", page_icon="🤖")

# --- 1. PASSWORD GATE ---
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.title("🔒 Internal Knowledge Base")
    st.caption("Enter your team password to access the AI Document Assistant.")
    password = st.text_input("Password", type="password")
    if st.button("Log In"):
        if password == APP_PASSWORD:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Incorrect password.")
    st.stop()

# --- 2. INITIALIZE PINECONE & OPENAI ---
@st.cache_resource
def init_rag():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    # Swap to "gpt-4o" for client production
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    return vector_store, retriever, llm

vector_store, retriever, llm = init_rag()

# --- 3. SIDEBAR: PDF UPLOADER & CONTROLS ---
with st.sidebar:
    st.header("📂 Knowledge Base")
    uploaded_files = st.file_uploader(
        "Upload new PDFs",
        type=["pdf"],
        accept_multiple_files=True
    )

    if st.button("Process & Upload") and uploaded_files:
        with st.spinner("Vectorizing documents..."):
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            total_chunks = 0

            for uploaded_file in uploaded_files:
                # Save to a temporary file so PyPDFLoader can read it
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = tmp.name

                try:
                    loader = PyPDFLoader(tmp_path)
                    docs = loader.load()

                    # Override temp path with the real filename for clean citations
                    for doc in docs:
                        doc.metadata["source"] = uploaded_file.name

                    chunks = text_splitter.split_documents(docs)
                    vector_store.add_documents(chunks)
                    total_chunks += len(chunks)
                finally:
                    os.remove(tmp_path)

            st.success(f"Added {total_chunks} chunks from {len(uploaded_files)} file(s).")

    st.divider()
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

    if st.button("Log Out"):
        st.session_state.authenticated = False
        st.rerun()

# --- 4. MAIN CHAT INTERFACE ---
st.title("🤖 AI Document Assistant")
st.caption("Powered by RAG • Strictly grounded in your uploaded documents")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"]:
            with st.expander("📄 View Sources"):
                for src in msg["sources"]:
                    st.write(f"- {src}")

if prompt := st.chat_input("Ask a question about your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents..."):
            docs = retriever.invoke(prompt)

            context = "\n\n".join([doc.page_content for doc in docs])
            sources = list(set([
                f"**{doc.metadata.get('source', 'Unknown')}** (Page {int(doc.metadata.get('page', 0)) + 1})"
                for doc in docs
            ]))

            system_prompt = (
                "You are a precise B2B document assistant. Answer the user's question "
                "using ONLY the provided context below. If the answer is not explicitly "
                "stated in the context, respond with: 'I cannot find that information "
                "in the provided documents.' Do not guess or use outside knowledge.\n\n"
                f"CONTEXT:\n{context}"
            )

            response = llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=prompt)
            ])
            answer = response.content

            st.markdown(answer)
            if "cannot find that information" not in answer.lower():
                with st.expander("📄 View Sources"):
                    for src in sources:
                        st.write(f"- {src}")
            else:
                sources = []

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources
    })