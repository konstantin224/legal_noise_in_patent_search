

import gc
import faiss
import torch

import numpy as np
import pandas as pd


from tqdm import tqdm
from typing import List
from sentence_transformers import SentenceTransformer



def safe_encode(model, texts: List[str], initial_batch_size: int = 1, max_oom_retries: int = 3) -> np.ndarray:
        """Кодирует тексты с автоматическим восстановлением после OOM."""
        batch_size = initial_batch_size
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        for attempt in range(max_oom_retries):
            try:
                with torch.inference_mode():
                    return model.encode(
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
                   
                    torch.cuda.empty_cache()
                    gc.collect()
                    
                    new_batch = max(1, batch_size // 2)
                    print(f"⚠️ OOM (попытка {attempt+1}/{max_oom_retries}) | Batch: {batch_size} → {new_batch}")
                    batch_size = new_batch
                else:
                    raise
        
        print("📉 Fallback на CPU (будет медленнее, но надёжно)")
        return model.encode(
            texts,
            batch_size=max(1, initial_batch_size // 4),
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
            device="cpu"
        )   


def main():

    model_name = "KaLM-Embedding/KaLM-embedding-multilingual-mini-instruct-v2.5"
    print(f"🚀 Загрузка модели: {model_name}")
    model = SentenceTransformer(model_name, device='cuda')
    
    dimension = model.get_embedding_dimension()

    encode_batch_size = 4

    for mode in ['claims', 'description']:
        
        index = faiss.IndexFlatL2(dimension)
        print(f'Делаем векторные представления для {mode}')
        
        all_vectors = [] # Временное хранилище для всех векторов режима

        # Проходим по всем файлам
        for i in range(10):
            path_to_csv = rf"C:\Users\rudkevich-k\Desktop\Обработка патентов\dataset_csv\dataset_for_test_fips_{i+1}.csv"
            
            try:
                # Читаем только нужную колонку, чтобы экономить память
                df_part = pd.read_csv(path_to_csv, index_col=0)
                
                # Очищаем от NaN, иначе модель упадет
                texts = df_part[mode].dropna().astype(str).tolist()
                
                if not texts:
                    continue

                print(f"   📄 Файл {i+1}: Обработка {len(texts)} текстов...")

                # 🔥 ГЛАВНОЕ ИСПРАВЛЕНИЕ: Кодируем ВСЕ тексты файла сразу (или большими кусками)
                # model.encode сам разобьет их на батчи внутри
                embeddings = safe_encode(model, texts, encode_batch_size, 1)
                
                # Добавляем в общий список
                all_vectors.append(embeddings)
                
                # Очистка памяти после большого файла
                del df_part, texts, embeddings
                gc.collect()
                torch.cuda.empty_cache()

            except Exception as e:
                print(f"   ❌ Ошибка в файле {i+1}: {e}")


        if all_vectors:
            print("   🏗️ Сборка индекса FAISS...")
            final_embeddings = np.concatenate(all_vectors, axis=0).astype('float32')
            
            # Проверка на пустоту
            if final_embeddings.size > 0:
                index.add(final_embeddings)
                print(f"   ✅ Добавлено векторов: {index.ntotal}")
            else:
                print("   ⚠️ Векторы не были добавлены (пустой датасет?)")
            
            del all_vectors, final_embeddings
            gc.collect()
        
        faiss.write_index(index, fr"C:\Users\rudkevich-k\Desktop\Обработка патентов\pre_compute_embeddings_for_search\indexs\{mode}_index.bin")

        del index
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    try:
        main()
        print("\n✅ ВСЕ ГОТОВО! Индексы сохранены.")
    except Exception as e:
        print(f"\n💥 КРИТИЧЕСКАЯ ОШИБКА: {e}")
        import traceback
        traceback.print_exc() # Это напечатает подробный стек ошибки
        input("Нажмите Enter, чтобы выйти...") # Чтобы окно не закрылось сразу