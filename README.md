# Advanced Multi-PDF RAG System using Qdrant and Hugging Face

## Overview

This project implements a Retrieval-Augmented Generation (RAG) system that enables users to upload multiple PDF documents, perform semantic search across them, and generate structured outputs such as answers and multiple-choice questions (MCQs).

The system combines vector search using Qdrant with large language models from Hugging Face to produce context-aware, grounded responses with source citations.


## Key Features

- Multi-PDF upload and persistent storage  
- Semantic search using dense vector embeddings  
- Filtering by specific PDF documents  
- Checkbox-based document selection  
- MCQ generation from document content  
- Source citations including document name and page number  
- Reset functionality to clear stored data and documents  
- Local storage of uploaded PDFs  


## System Architecture

PDF Documents  
↓  
**Text Extraction** (PyMuPDF)  
↓  
**Embedding Generation** (Sentence-Transformers)  
↓  
**Qdrant Vector Database**  
↓  
**User Query** → Embedding  
↓  
**Similarity Search** (Top-K Retrieval)  
↓  
**Context Construction**  
↓  
**Hugging Face LLM** (Gemma)  
↓  
**Final Output** (Answers / MCQs with Citations)


## Technology Stack

- Frontend: Streamlit  
- Backend: Python 3.10+  
- Vector Database: Qdrant Cloud  
- Embedding Model: sentence-transformers/all-MiniLM-L6-v2  
- Language Model: google/gemma-2-2b-it (Hugging Face Inference API)  
- PDF Processing: PyMuPDF  
- Environment Management: python-dotenv  


## Project Structure

```text
rag-qdrant/
├── uploaded_pdfs/    # Directory for stored PDF documents
├── .env              # Environment variables (API keys, URLs)
├── ingest.py         # PDF processing and vector indexing logic
├── search.py         # Retrieval logic and LLM interaction
├── ui.py             # Streamlit-based user interface
├── requirements.txt  # Project dependencies
└── README.md         # Project documentation
```

## Environment Setup

**Note:** You must have a Qdrant Cloud account and a Hugging Face API token to run this project.

Create a `.env` file in the root directory with the following variables:


```env
HF_TOKEN=your_huggingface_token
QDRANT_URL=https://YOUR_CLUSTER_ID.us-east4-0.gcp.cloud.qdrant.io
QDRANT_API_KEY=your_qdrant_api_key
COLLECTION_NAME=docs
```

## Installation

### Step 1: Create and Activate Virtual Environment

```bash
# Create the environment
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# Activate on macOS/Linux
source venv/bin/activate
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```
**Torch Installation (CPU Only)**  
If the `torch` installation fails or if you want a lightweight version for systems without a GPU, use:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### Step 3: Run the Application

```bash
streamlit run ui.py
```

The application will be accessible at `http://localhost:8501`.


## Usage

1. Upload one or more PDF documents  
2. Wait for indexing to complete  
3. Optionally select specific PDFs for filtering  
4. Enter a query or request MCQs  
5. View generated output along with citations  
6. Use the reset option to clear stored data  


## Example Queries

- Summarize the uploaded documents  
- Generate 10 MCQs from the content  
- Explain a specific concept from a selected PDF  
- Compare topics across multiple PDFs  


## Design Highlights

- Uses dense vector representations for semantic understanding  
- Ensures grounded responses by restricting LLM output to retrieved context  
- Supports metadata-based filtering for targeted retrieval  
- Provides transparency through document-level citations  


## Use Cases

- Academic preparation and revision  
- Document-based question answering  
- Corporate knowledge base systems  
- Training and educational tools  
