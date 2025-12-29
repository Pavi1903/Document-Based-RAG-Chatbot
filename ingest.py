import fitz
import uuid
import time
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

load_dotenv()

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
)

COLLECTION = os.getenv("COLLECTION_NAME")

def ensure_collection():
    collections = [c.name for c in client.get_collections().collections]
    if COLLECTION not in collections:
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(
                size=384,
                distance=Distance.COSINE
            )
        )

def ingest(pdf_path: str, pdf_name: str):
    ensure_collection()

    pdf = fitz.open(pdf_path)
    points = []

    for page_number, page in enumerate(pdf, start=1):
        text = page.get_text()
        if not text.strip():
            continue

        vector = model.encode(text).tolist()

        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={
                    "text": text,
                    "source": pdf_name,
                    "page": page_number
                }
            )
        )

    # Batch upload (safe)
    for i in range(0, len(points), 10):
        client.upsert(
            collection_name=COLLECTION,
            points=points[i:i+10]
        )
        time.sleep(1)
