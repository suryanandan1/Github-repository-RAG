import chromadb

COLLECTION_NAME = (
    "github_repository"
)


class VectorStore:

    def __init__(self):

        self.client = (
            chromadb.PersistentClient(
                path="chroma_db"
            )
        )

        self.collection = (
            self.client
            .get_or_create_collection(
                name=COLLECTION_NAME
            )
        )

    def add_chunk(
        self,
        chunk_id,
        text,
        embedding,
        metadata
    ):

        self.collection.add(
            ids=[chunk_id],
            documents=[text],
            embeddings=[embedding],
            metadatas=[metadata]
        )

    def count(self):

        return (
            self.collection.count()
        )

    def search(
        self,
        query_embedding,
        n_results=5
    ):

        return (
            self.collection.query(
                query_embeddings=[
                    query_embedding
                ],
                n_results=n_results
            )
        )

    def clear_collection(self):

        try:

            self.client.delete_collection(
                COLLECTION_NAME
            )

        except Exception:
            pass

        self.collection = (
            self.client
            .get_or_create_collection(
                name=COLLECTION_NAME
            )
        )

    def reset(self):

        self.clear_collection()

        return (
            self.collection.count()
        )