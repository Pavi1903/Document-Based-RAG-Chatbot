import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchAny
from huggingface_hub import InferenceClient

load_dotenv()

# Embedding model
embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Qdrant client
client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
)

# Hugging Face chat model
llm = InferenceClient(
    model="google/gemma-2-2b-it",
    token=os.getenv("HF_TOKEN")
)

COLLECTION = os.getenv("COLLECTION_NAME")

# 🔹 NEW: Fetch list of PDFs from Qdrant
def get_available_pdfs():
    points, _ = client.scroll(
        collection_name=COLLECTION,
        limit=1000,
        with_payload=True
    )
    return sorted(set(p.payload["source"] for p in points))


def ask(question: str, selected_pdfs: list):
    # 1️⃣ Embed question
    query_vector = embed_model.encode(question).tolist()

    # 2️⃣ Apply PDF filter if selected
    qdrant_filter = None
    if selected_pdfs:
        qdrant_filter = Filter(
            must=[
                FieldCondition(
                    key="source",
                    match=MatchAny(any=selected_pdfs)
                )
            ]
        )

    # 3️⃣ Search Qdrant
    hits = client.search(
        collection_name=COLLECTION,
        query_vector=query_vector,
        limit=8,
        query_filter=qdrant_filter,
        with_payload=True
    )

    # 4️⃣ Build context
    context = "\n\n".join(hit.payload["text"] for hit in hits)

    # 5️⃣ Citations
    citations = "\n".join(
        f"- {hit.payload['source']} (Page {hit.payload['page']})"
        for hit in hits
    )

    # 6️⃣ Prompt
    messages = [
        {
            "role": "system",
            "content": (
                "Create EXACTLY 10 multiple-choice questions (MCQs). "
                "Number them from Question 1 to Question 10. "
                "Each question must have 4 options (A–D). "
                "Clearly mention the correct answer. "
                "Use ONLY the provided context."
            )
        },
        {
            "role": "user",
            "content": f"""
Context:
{context}

Task:
{question}
"""
        }
    ]

    # 7️⃣ Generate answer
    response = llm.chat_completion(
        messages=messages,
        max_tokens=900
    )

    return response.choices[0].message.content, citations
