import chromadb


class VectorStore:

    def __init__(self):

        self.client = chromadb.PersistentClient(
            path="chroma_db"
        )

        self.collection = (
            self.client.get_or_create_collection(
                name="github_repository"
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

        return self.collection.count()

    def search(
        self,
        query_embedding,
        n_results=5
    ):

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )

        return results