from src.vector_store import VectorStore
from src.mistral_embeddings import generate_embedding

def retrieve_context(
    question,
    top_k=5
):

    vector_store = VectorStore()

    query_embedding = generate_embedding(
        question
    )

    results = vector_store.search(
        query_embedding,
        n_results=top_k
    )

    documents = (
        results["documents"][0]
    )

    return "\n\n".join(
        documents
    )