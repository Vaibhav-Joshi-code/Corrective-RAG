"""Persistent Qdrant retrieval; Azure still creates the embeddings."""
import os
from uuid import NAMESPACE_URL, uuid5

from langchain_core.documents import Document
from qdrant_client import QdrantClient, models


class QdrantStore:
    """Small adapter keeping the graph's similarity_search interface unchanged."""

    def __init__(self, client, embeddings, collection_name):
        self.client = client
        self.embeddings = embeddings
        self.collection_name = collection_name
        self.point_ids = []

    def add_documents(self, documents):
        """Upsert stable IDs; never recreate or delete an existing collection."""
        if not documents:
            raise ValueError("Provide at least one knowledge-base document.")
        ids = [doc.metadata.get("id") for doc in documents]
        if any(not isinstance(value, str) or not value for value in ids):
            raise ValueError("Each document needs a non-empty string metadata.id.")
        if len(ids) != len(set(ids)):
            raise ValueError("Knowledge-base document IDs must be unique.")

        vectors = self.embeddings.embed_documents([doc.page_content for doc in documents])
        if len(vectors) != len(documents) or not vectors or not vectors[0]:
            raise ValueError("The embedding provider returned an invalid batch.")
        dimension = len(vectors[0])
        if any(len(vector) != dimension for vector in vectors):
            raise ValueError("All document embeddings must have the same dimension.")

        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(size=dimension, distance=models.Distance.COSINE),
            )
        else:
            config = self.client.get_collection(self.collection_name).config.params.vectors
            if not isinstance(config, models.VectorParams) or (
                config.size != dimension or config.distance != models.Distance.COSINE
            ):
                raise ValueError(
                    "Qdrant collection must use an unnamed cosine vector with "
                    f"dimension {dimension}. Choose a new QDRANT_COLLECTION_NAME "
                    "for this embedding deployment; existing data was not deleted."
                )

        point_ids = [str(uuid5(NAMESPACE_URL, f"corrective-rag-crag-agent/{doc_id}")) for doc_id in ids]
        points = [
            models.PointStruct(
                id=point_id, vector=vector,
                payload={"page_content": doc.page_content, "metadata": doc.metadata},
            )
            for point_id, doc, vector in zip(point_ids, documents, vectors)
        ]
        self.client.upsert(collection_name=self.collection_name, points=points, wait=True)
        self.point_ids = point_ids

    def similarity_search(self, query, k=4):
        """Embed the query, search Qdrant, and reconstruct LangChain Documents."""
        if not self.point_ids:
            raise ValueError("Call add_documents before searching.")
        if k < 1:
            raise ValueError("k must be positive.")
        result = self.client.query_points(
            collection_name=self.collection_name,
            query=self.embeddings.embed_query(query),
            # Only this run's knowledge-base IDs: ignore unrelated or removed entries.
            query_filter=models.Filter(must=[models.HasIdCondition(has_id=self.point_ids)]),
            limit=k, with_payload=True, with_vectors=False,
        )
        return [
            Document(page_content=point.payload["page_content"], metadata=point.payload["metadata"])
            for point in result.points
        ]


def build_vector_store(embeddings, documents):
    """Connect to configured Qdrant and seed/update this lesson's documents."""
    url = (os.getenv("QDRANT_URL") or os.getenv("qdrant_url") or "").strip()
    api_key = (os.getenv("QDRANT_API_KEY") or os.getenv("qdrant_api_key") or "").strip()
    if not url:
        raise ValueError("Set QDRANT_URL in .env before starting the server.")
    collection = os.getenv("QDRANT_COLLECTION_NAME", "corrective_rag_crag").strip()
    if not collection:
        raise ValueError("QDRANT_COLLECTION_NAME must not be blank.")
    client = QdrantClient(url=url, api_key=api_key or None, timeout=30)
    store = QdrantStore(client, embeddings, collection)
    store.add_documents(documents)
    return store
