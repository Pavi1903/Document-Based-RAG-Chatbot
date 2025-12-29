import streamlit as st
import os
from ingest import ingest
from search import ask, get_available_pdfs
from qdrant_client import QdrantClient
from dotenv import load_dotenv
import shutil

load_dotenv()

UPLOAD_DIR = "uploaded_pdfs"
os.makedirs(UPLOAD_DIR, exist_ok=True)

client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
)

COLLECTION = os.getenv("COLLECTION_NAME")

st.set_page_config(page_title="Qdrant RAG", page_icon="📄")
st.title("📄 Qdrant RAG")

# 🔥 RESET BUTTON
if st.button("🗑️ Delete / Reset All PDFs"):
    client.delete_collection(COLLECTION)

    # Optional: also delete stored PDFs
    shutil.rmtree(UPLOAD_DIR)
    os.makedirs(UPLOAD_DIR, exist_ok=True)

    st.success("Collection and uploaded PDFs reset successfully!")
    st.stop()

# 📥 Upload PDFs
uploaded_files = st.file_uploader(
    "Upload one or more PDFs",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_files:
    st.info("Indexing PDFs…")

    for file in uploaded_files:
        save_path = os.path.join(UPLOAD_DIR, file.name)

        # Avoid re-saving the same PDF
        if not os.path.exists(save_path):
            with open(save_path, "wb") as f:
                f.write(file.getvalue())

            ingest(save_path, file.name)

    st.success("PDFs indexed and saved successfully!")

# 📚 Show uploaded PDFs
st.subheader("📁 Uploaded PDFs")
existing_pdfs = os.listdir(UPLOAD_DIR)
if existing_pdfs:
    for pdf in existing_pdfs:
        st.write(f"• {pdf}")
else:
    st.write("No PDFs uploaded yet.")

# 📚 PDF FILTER CHECKBOXES
available_pdfs = []
try:
    available_pdfs = get_available_pdfs()
except:
    pass

selected_pdfs = st.multiselect(
    "Filter by PDF (optional)",
    available_pdfs,
    default=available_pdfs
)

# ❓ Ask question
question = st.text_input("Ask a question or request MCQs")

if question:
    answer, citations = ask(question, selected_pdfs)

    st.subheader("📘 Answer")
    st.write(answer)

    st.subheader("📌 Citations")
    st.write(citations)
