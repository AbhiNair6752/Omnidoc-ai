from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

COLLECTION_NAME = "omnidoc_policies"

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

client = QdrantClient(
    host="localhost",
    port=6333
)

def retrieve_relevant_chunks(
        query: str,
        limit: int = 3
):

    query_embedding = embedding_model.encode(
        query
    )

    response = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_embedding.tolist(),
        limit=limit
    )

    results = response.points

    chunks = []

    for result in results:

        chunks.append({
            "text": result.payload["text"],
            "score": result.score,
            "document_type": result.payload.get("document_type")
        })

    return chunks


if __name__=="__main__":

    query = (
         "What should happen if an invoice "
        "is missing important information?"
    )

    results = retrieve_relevant_chunks(
        query
    )

    for result in results:

        print("\n----Retrieved Chunk----")

        print(
            result["text"]
        )

        print(
            f"score: {result['score']}"
        )