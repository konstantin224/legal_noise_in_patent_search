from system_path import *

from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer


def get_model_embeddings_and_client_qdrant():
    client = QdrantClient(host="localhost", port=6335)
    model = SentenceTransformer(PATH_TO_EMBED_MODEL, device = 'cuda', cache_folder= CACHE_FOLDER_PATH)

    if not client.collection_exists(COLLECTION):
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=models.VectorParams(
                size=model.get_sentence_embedding_dimension(),
                distance=models.Distance.COSINE,
                on_disk=True
            )
        )
    
    return client, model