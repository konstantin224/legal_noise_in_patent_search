
import gc
import torch

import numpy as np

from typing import List
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer


class QdrantWorking:

    def __init__(self, client, collection):

        self.client = client
        self.collection = collection
        self.model = None
        # SentenceTransformer(r"C:\Users\rudkevich-k\.cache\huggingface\hub\models--KaLM-Embedding--KaLM-embedding-multilingual-mini-instruct-v2.5", device = 'cuda')




    def safe_encode(self, texts: List[str], initial_batch_size: int = 1, max_oom_retries: int = 1) -> np.ndarray:
        """Кодирует тексты с автоматическим восстановлением после OOM."""
        batch_size = initial_batch_size
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        for attempt in range(max_oom_retries):
            try:
                with torch.inference_mode():
                    return self.model.encode(
                        texts,
                        batch_size=batch_size,
                        normalize_embeddings=True,
                        convert_to_numpy=True,
                        show_progress_bar=False,
                        device=device
                    )
            except RuntimeError as e:
                err_msg = str(e).lower()
                if "out of memory" in err_msg and device == "cuda":
                    # 🔥 Чистим память
                    torch.cuda.empty_cache()
                    gc.collect()
                    
                    # Уменьшаем батч
                    new_batch = max(1, batch_size // 2)
                    print(f"⚠️ OOM (попытка {attempt+1}/{max_oom_retries}) | Batch: {batch_size} → {new_batch}")
                    batch_size = new_batch
                else:
                    raise
        
        print("📉 Fallback на CPU (будет медленнее, но надёжно)")
        return self.model.encode(
            texts,
            batch_size=max(1, initial_batch_size // 4),
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
            device="cpu"
        )   

    


    def doc_exists(self, doc_id: str) -> bool:

        result = self.client.retrieve(
            collection_name = self.collection,
            ids=[int(doc_id)],
            with_payload=False,  
            with_vectors=False
        )
        return len(result) > 0

    def search_docs(self, transform_query_vector, query_vec: str = None, query: str = None, top_k: int = 31):


        if query_vec is not None:
             
            query_vec = transform_query_vector(query_vec)

            hits = self.client.query_points(
                collection_name=self.collection,
                query=query_vec,
                limit=top_k,
                with_payload=False
        )

        else:

            query_vec = self.safe_encode(query).tolist()

            query_vec = transform_query_vector(query_vec)
            
            
            hits = self.client.query_points(
                    collection_name=self.collection,
                    query=query_vec,
                    limit=top_k,
                    with_payload=False
            )

        del query_vec

        return [{"id": h.id, "score": h.score, "payload": h.payload} for h in hits.points]
