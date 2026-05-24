import os
import shutil
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ---------------------------------------------------------
# Load environment variables
# ---------------------------------------------------------
load_dotenv()


# ---------------------------------------------------------
# Admin settings
# ---------------------------------------------------------
INDEX_DIR = "insurance_faiss_index"
EMBEDDING_MODEL = "models/gemini-embedding-001"


# ---------------------------------------------------------
# Streamlit page setup
# ---------------------------------------------------------
st.title("SecureLife Admin Portal 🛠️")

st.caption(
    "Manage the insurance knowledge base used by SecureLife Assist. "
    "Upload approved policy documents, FAQ files, claims guides, and service information."
)


# ---------------------------------------------------------
# Helper functions
# ---------------------------------------------------------
def api_key_exists() -> bool:
    """Check if the Gemini API key exists."""
    return bool(os.getenv("GOOGLE_API_KEY"))


def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    """Create Gemini embeddings for FAISS."""
    return GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
    )


def read_uploaded_files(uploaded_files) -> list[Document]:
    """
    Read uploaded TXT and CSV files and convert them into LangChain Document objects.
    Each document keeps the original filename as metadata.
    """
    documents = []

    for uploaded_file in uploaded_files:
        file_name = uploaded_file.name
        file_extension = Path(file_name).suffix.lower()

        raw_bytes = uploaded_file.getvalue()

        try:
            text = raw_bytes.decode("utf-8")
        except UnicodeDecodeError:
            text = raw_bytes.decode("latin-1")

        if not text.strip():
            continue

        documents.append(
            Document(
                page_content=text,
                metadata={
                    "source": file_name,
                    "file_type": file_extension,
                },
            )
        )

    return documents


def delete_knowledge_base() -> None:
    """Delete the current FAISS index folder."""
    if os.path.isdir(INDEX_DIR):
        shutil.rmtree(INDEX_DIR)


# ---------------------------------------------------------
# Security checks
# ---------------------------------------------------------
if not api_key_exists():
    st.error("GOOGLE_API_KEY is missing. Add it to your .env file.")
    st.stop()


# ---------------------------------------------------------
# Knowledge base status
# ---------------------------------------------------------
st.divider()

st.subheader("Knowledge Base Status")

if os.path.isdir(INDEX_DIR):
    st.success("The SecureLife knowledge base is active and ready for customer questions.")
else:
    st.warning("No active knowledge base found. Upload documents to activate SecureLife Assist.")


col1, col2 = st.columns([1, 3])

with col1:
    delete_clicked = st.button(
        "Delete Current Knowledge Base",
        use_container_width=True,
    )

with col2:
    st.write(
        "Use this only when you want to replace old policy documents with a new approved version."
    )

if delete_clicked:
    if os.path.isdir(INDEX_DIR):
        delete_knowledge_base()
        st.success("The current knowledge base has been deleted.")
        st.rerun()
    else:
        st.info("There is no knowledge base to delete.")


# ---------------------------------------------------------
# Document upload
# ---------------------------------------------------------
st.divider()

st.subheader("Upload Approved Company Documents")

st.write(
    "Upload SecureLife policy documents, claims guides, FAQ files, coverage tables, "
    "renewal instructions, cancellation rules, complaint procedures, or branch service information."
)

uploaded_files = st.file_uploader(
    "Upload policy documents, claims guides, FAQs, or service files",
    type=["txt", "csv"],
    accept_multiple_files=True,
)

if uploaded_files:
    st.write("Selected files:")
    for uploaded_file in uploaded_files:
        st.write(f"- {uploaded_file.name}")


# ---------------------------------------------------------
# Build knowledge base
# ---------------------------------------------------------
publish_clicked = st.button(
    "Publish Knowledge Base",
    use_container_width=True,
)

if publish_clicked:
    if not uploaded_files:
        st.warning("Upload at least one approved company document first.")
        st.stop()

    try:
        with st.spinner("Reading uploaded company documents..."):
            documents = read_uploaded_files(uploaded_files)

        if not documents:
            st.error("No readable text was found in the uploaded files.")
            st.stop()

        with st.spinner("Splitting documents into searchable policy sections..."):
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=150,
                separators=["\n\n", "\n", ".", ",", " "],
            )

            docs = text_splitter.split_documents(documents)

        with st.spinner("Creating secure search index for SecureLife Assist..."):
            vectorstore = FAISS.from_documents(
                documents=docs,
                embedding=get_embeddings(),
            )

            vectorstore.save_local(INDEX_DIR)

        st.success(
            f"Knowledge base published successfully from {len(uploaded_files)} file(s). "
            f"{len(docs)} searchable document sections were created."
        )

        st.subheader("Published Files")
        for uploaded_file in uploaded_files:
            st.write(f"- {uploaded_file.name}")

        st.info(
            "Return to the SecureLife Assist chatbot page to test customer questions "
            "using the latest approved documents."
        )

    except Exception as exc:
        st.error(f"Failed to publish the knowledge base: {exc}")


# ---------------------------------------------------------
# Admin guidance
# ---------------------------------------------------------
st.divider()

st.info(
    "Admin reminder: only upload approved public-facing or internal support documents. "
)