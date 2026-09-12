from pathlib import Path
from sentence_transformers import SentenceTransformer

from qdrant_client import QdrantClient

from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct
)

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

client = QdrantClient(
    host="localhost",
    port=6333
)

COLLECTION_NAME = "omnidoc_policies"

BASE_DIR = Path(__file__).resolve().parents[2]

POLICY_FILE = (
    BASE_DIR
    / "knowledge_base"
    / "invoice"
    / "invoice_policy.txt"
)


def load_policy():

    with open(
        POLICY_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        return file.read()

def chunk_policy(text: str):

    chunks = [
        chunk.strip()
        for chunk in text.split("\n\n")
        if chunk.strip()
    
    ]

    return chunks

def create_collection():

    if client.collection_exists(
        COLLECTION_NAME
    ):

       client.delete_collection(
           COLLECTION_NAME
       )

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=384,
            distance=Distance.COSINE
        )
    )

    print(
        f"Collection '{COLLECTION_NAME}' created"
    )

def ingest_policy(chunks):

    points = []

    for index, chunk in enumerate(chunks):

        embedding = embedding_model.encode(
            chunk
        )

        point = PointStruct(
            id=index,
            vector=embedding.tolist(),
            payload={
                "text": chunk,
                "document_type": "invoice",
                "source": "invoice_policy.txt"
            }
        )

        points.append(point)

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points
    )

    print(
        f"{len(points)} policy chunks inserted into Qdrant"
    )


if __name__ == "__main__":

    policy = load_policy()

    chunks = chunk_policy(policy)

    print(f"Number of chunks {len(chunks)}")

    for i , chunk in enumerate(chunks):

        print(f"\n---Chunk {i}---")
        print(chunk)

    create_collection()

    ingest_policy(chunks)