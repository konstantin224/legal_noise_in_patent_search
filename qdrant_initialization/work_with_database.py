
import gc
import re
import torch
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET


from models import *
from system_path import *
from typing import List, Dict
from qdrant_client import models

# from __future__ import annotations



client, model = get_model_embeddings_and_client_qdrant()

OFFICES = r'(?:WO|US|EP|RU|CN|JP|KR|DE|GB|FR|CA|AU|IN|BR|MX|ES|IT|PL|UA|BY|KZ|UZ)'
PATENT_RE = re.compile(rf'\b({OFFICES})\s*([\d][A-Za-z0-9/\-]*)', re.IGNORECASE)


def parse_patent_citations(text: str) -> List[Dict[str, str]]:
    if not text or not isinstance(text, str):
        return []

    results = []
    for match in PATENT_RE.finditer(text):
        office = match.group(1).upper()
        number = match.group(2).upper()
        
        # Убираем возможные артефакты с краёв
        number = number.rstrip('.,;:!?()[]{}"\'')
        
        # Настоящие патентные номера обычно >= 4 символов
        if len(number) < 4:
            continue
            
        results.append({
            "office": office,
            "number": number,
            "full_id": f"{office}{number}"  # Готовый ключ для БД
        })
        
    return results


def doc_exists(client, collection_name: str, doc_id: str) -> bool:
    """Проверяет, существует ли точка с таким ID в коллекции"""
    result = client.retrieve(
        collection_name=collection_name,
        ids=[int(doc_id)],
        with_payload=False,  # Не тянем лишние данные, только проверка существования
        with_vectors=False
    )
    return len(result) > 0

def assert_patent_citation(cit_list):

    count = 0

    for item in cit_list:
        get_info_from_db = doc_exists(client, 'patents', item['number'])
        
        if get_info_from_db:
            count += 1
    if count >= 2:
        return True
    else:
        return False
    
def safe_encode(model, texts: List[str], initial_batch_size: int = 8, max_oom_retries: int = 3) -> np.ndarray:
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
                # 🔥 Чистим память
                torch.cuda.empty_cache()
                gc.collect()
                
                # Уменьшаем батч
                new_batch = max(1, batch_size // 2)
                print(f"⚠️ OOM (попытка {attempt+1}/{max_oom_retries}) | Batch: {batch_size} → {new_batch}")
                batch_size = new_batch
            else:
                # Если ошибка не связана с памятью → пробрасываем дальше
                raise
    
    # 🔄 Если GPU не справился → последний шанс на CPU
    print("📉 Fallback на CPU (будет медленнее, но надёжно)")
    return model.encode(
        texts,
        batch_size=max(1, initial_batch_size // 4),
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
        device="cpu"
    )   

# 2. Индексация
def index_docs_fast(df: pd.DataFrame, test_dataset: pd.DataFrame = None) -> pd.DataFrame:
    """
    Быстрая индексация чанка DataFrame в Qdrant.
    Возвращает обновленный test_dataset (если сработало условие >=2 цитат в БД).
    """
    if df.empty:
        return test_dataset if test_dataset is not None else pd.DataFrame()

    # 🔹 1. Предобработка (без копирования лишних данных)
    df_work = df.copy()
    # df_work['pub_id'] = df_work['publication_number'].astype(str).str.strip()
    df_work['embed_text'] = (df_work['title'].fillna('') + ' ' + df_work['claims'].fillna('')).str.strip()
    
    df_work['pub_id'] = pd.to_numeric(df_work['publication_number'], errors='coerce').astype('Int64')
    df_work = df_work.dropna(subset=['pub_id'])  # Убираем строки, где ID не распозналось
    df_work['pub_id'] = df_work['pub_id'].astype(int)


    # Фильтруем строки с текстом для эмбеддинга
    valid_mask = df_work['embed_text'].str.len() > 10
    valid_idx = df_work.index[valid_mask]
    
    # 🔹 2. ПАКЕТНОЕ кодирование (главная оптимизация скорости)
    vectors_map = {}  # {idx: vector_list}
    
    torch.cuda.empty_cache()
    gc.collect()
    
    if len(valid_idx) > 0:
        texts = df_work.loc[valid_idx, 'embed_text'].tolist()
        
        with torch.inference_mode():
            vectors = safe_encode(model, texts, initial_batch_size=8)
        # print("ебанули вектора для текстов")
        # Сохраняем векторы в словарь по индексу строки
        for idx, vec in zip(valid_idx, vectors.tolist()):
            vectors_map[idx] = vec

        del vectors, texts
        gc.collect()
    else:
        return test_dataset if test_dataset is not None else pd.DataFrame()

    # 🔹 3. ПАКЕТНАЯ проверка существования в БД
    # Проверяем текущие ID + все упомянутые в цитатах (чтобы не дёргать сеть построчно)
    current_ids = df_work['pub_id'].tolist()
    
    # Собираем все cited ID из текущего чанка
    cited_map = {}  # row_index -> list[cited_ids]
    all_cited_ids = set()
    for idx, row in df_work.iterrows():
        cits = parse_patent_citations(str(row.get('citations', '')))
        
        cited_ids = []
        for c in cits:
            try:
                cited_ids.append(int(c['number']))
            except:
                pass

        # cited_ids = [int(c['number']) for c in cits]
        cited_map[idx] = cited_ids
        all_cited_ids.update(cited_ids)
        
    ids_to_check = list(set(current_ids) | all_cited_ids)
    
    # Один запрос к Qdrant на все проверки


    ids_to_check = list(map(int, ids_to_check))

    # print(ids_to_check)

    existing_points = client.retrieve(COLLECTION, ids=ids_to_check, with_payload=False, with_vectors=False)
    existing_set = {str(p.id) for p in existing_points}

    # 🔹 4. Сборка точек и фильтрация для test_dataset
    points = []
    test_rows = []

    for idx in valid_idx:
        pid = int(str(df_work.loc[idx, 'pub_id']).strip())
        
        # Пропускаем, если уже в БД
        if pid in existing_set:
            continue
            
        vec = vectors_map.get(idx)
        if not vec or not isinstance(vec, list):
            continue


        # Формируем payload (только нужное)
        payload = {
            "publication_number": pid,
            "title": df_work.loc[idx, 'title'],
            "publication_date": df_work.loc[idx, 'publication_date'],
            'abstract': df_work.loc[idx, 'abstract'],
            'claims': df_work.loc[idx, 'claims'],
            'description': df_work.loc[idx, 'description'] 
        }
        points.append(models.PointStruct(id=pid, vector=vec, payload=payload))

        # Условие для test_dataset: >=2 цитаты уже есть в БД
        cited_count = sum(1 for c in cited_map.get(idx, []) if c in existing_set)
        if cited_count >= 2:
            test_rows.append(df.loc[idx].to_dict())

    # 🔹 5. Загрузка в Qdrant
    if points:
        client.upsert(collection_name=COLLECTION, points=points)
        
    # 🔹 6. Обновление test_dataset
    if test_rows:
        new_df = pd.DataFrame(test_rows)
        if test_dataset is not None and not test_dataset.empty:
            test_dataset = pd.concat([test_dataset, new_df], ignore_index=True)
        else:
            test_dataset = new_df

          
    del points, vectors_map, cited_map, df_work, valid_idx, existing_points, existing_set
    gc.collect()
    torch.cuda.empty_cache()

    return test_dataset
# 3. Поиск
def search_docs(query: str, top_k: int = 20):
    query_vec = model.encode(query, normalize_embeddings=True).tolist()
    
    hits = client.query_points(
        collection_name=COLLECTION,
        query=query_vec,
        limit=top_k,
        with_payload=True
    )
    
    return [{"id": h.id, "score": h.score, "payload": h.payload} for h in hits.points]