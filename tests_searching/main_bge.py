# Учесть, что нужно убирать из выдачи искомую заявку, извлекать top_k + 1


import gc
import os
import re
import sys
import torch
import faiss
import pickle
import argparse
import importlib.util

import numpy as np
import pandas as pd


from tqdm import tqdm
from config_py import *
from pathlib import Path
from qdrant_client import QdrantClient
from typing import List, Dict, Set, Union, Optional
from importlib.util import spec_from_file_location, module_from_spec

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from start_qdrant_for_working.start_qdrant import QdrantWorking


OFFICES = r'(?:WO|US|EP|RU|CN|JP|KR|DE|GB|FR|CA|AU|IN|BR|MX|ES|IT|PL|UA|BY|KZ|UZ)'
PATENT_RE = re.compile(rf'\b({OFFICES})\s*([\d][A-Za-z0-9/\-]*)', re.IGNORECASE)

def parse_patent_citations(text: str) -> List[Dict[str, str]]:
    if not text or not isinstance(text, str):
        return []

    results = []
    for match in PATENT_RE.finditer(text):
        office = match.group(1).upper()
        number = match.group(2).upper()
        
        number = number.rstrip('.,;:!?()[]{}"\'')
        

        if len(number) < 4:
            continue
            
        results.append({
            "office": office,
            "number": number,
            "full_id": f"{office}{number}"  
        })
        
    return results

def extract_gt(collection, client, list_of_json: List):

    ids_for_check = []

    for item in list_of_json:
        if item["office"] != 'RU':
            continue
        
        ids_for_check.append(item['number'])

    ids_for_check = list(map(int, ids_for_check))

    points = client.retrieve(
        collection_name=collection,
        ids=ids_for_check,
        with_payload=False,
        with_vectors=False  # Обычно вектор не нужен при чтении метаданных
        )
    
    gt_list = [p.id for p in points]
    
    return gt_list

def filter_query_doc(results: List[Dict], query_doc_id: str) -> List[Dict]:
    return [r for r in results if r.get('id') != query_doc_id]

def calculate_recall_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    """
    Recall@K = |Relevant ∩ Retrieved@K| / |Relevant|
    
    Доля релевантных документов, найденных в топ-K.
    """
    if not relevant:
        return 1.0  # Нет релевантных → идеальный recall по соглашению
    
    relevant = set(relevant)
    retrieved_at_k = set(retrieved[:k])
    relevant_found = len(retrieved_at_k & relevant)
    
    return relevant_found / len(relevant)


def calculate_precision_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    """
    Precision@K = |Relevant ∩ Retrieved@K| / K
    
    Доля релевантных документов среди топ-K результатов.
    """
    if k == 0:
        return 1.0
    
    relevant = set(relevant)
    retrieved_at_k = set(retrieved[:k])
    relevant_found = len(retrieved_at_k & relevant)
    
    return relevant_found / k


def calculate_average_precision(retrieved: List[str], relevant: List[str], k: Optional[int] = None) -> float:
    """
    Average Precision (AP) для одного запроса.
    
    AP = (1/|Relevant|) × Σ(P@i × rel(i)), где rel(i)=1 если документ релевантен
    
    Учитывает ранжирование: релевантные документы выше → выше AP.
    """
    relevant = set(relevant)

    if not relevant:
        return 1.0
    
    if k is not None:
        retrieved = retrieved[:k]
    
    if not retrieved:
        return 0.0
    
    ap_sum = 0.0
    relevant_found = 0
    
    for i, doc_id in enumerate(retrieved):
        if doc_id in relevant:
            relevant_found += 1
            precision_at_i = relevant_found / (i + 1)  # P@i
            ap_sum += precision_at_i
    
    return ap_sum / len(relevant)

def calculate_mrr(retrieved: List[str], relevant: List[str], k: int) -> float:
    """
    Mean Reciprocal Rank (для одного запроса — просто Reciprocal Rank).
    
    RR = 1 / rank_первого_релевантного
    """

    relevant = set(relevant)

    if not relevant:
        return 1.0
    
    for i, doc_id in enumerate(retrieved[:k]):
        if doc_id in relevant:
            return 1.0 / (i + 1)
    
    return 0.0  # Не найдено в топ-K


def load_experiment_module(experiment_name):
    """
    Динамически импортирует файл search_<experiment_name>.py
    из текущей директории.
    """
    # 1. Формируем имя файла
    filename = f"search_{experiment_name}.py"
    
    # 2. Определяем полный путь к файлу (в той же папке, где лежит этот скрипт)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(current_dir, filename)
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"❌ Файл эксперимента не найден: {filepath}")
    
    print(f"Загрузка модуля из: {filepath}")
 
    spec = importlib.util.spec_from_file_location(f"search_{experiment_name}", filepath)
    
    module = importlib.util.module_from_spec(spec)

    spec.loader.exec_module(module)
    
    return module


def save_checkpoint(filepath, results, last_index):
    """Сохраняет текущее состояние в файл"""
    data = {
        'results': results,
        'last_index': last_index
    }
    # Используем временный файл, чтобы не повредить основной при сбое во время записи
    temp_path = filepath + ".tmp"
    with open(temp_path, 'wb') as f:
        pickle.dump(data, f)
    os.replace(temp_path, filepath) 




def main():
     
    parser = argparse.ArgumentParser(description="Запуск экспериментов поиска")
    parser.add_argument("experiment", type=str, help="Название эксперимента (без префикса search_)")
    args = parser.parse_args()


    try:
        # 4. Импортируем нужный файл
        exp_module = load_experiment_module(args.experiment)
        
        transform_query_vector = exp_module.transform_query_vector
        
        print(f"✅ Модуль search_{args.experiment} успешно загружен.")
            
    except Exception as e:
        print(f"❌ Ошибка при запуске: {e}")


    mode_search = [
                   'claims',
                   'description'
                   ]

    # test_dataset = pd.read_csv(r"C:\Users\rudkevich-k\Desktop\Обработка патентов\dataset_for_test_fips.csv", 
    #                             usecols=["publication_number", "claims", "description", "citations"],
    #                             index_col = 0, 
    #                             engine="c",        # ← стандартный движок pandas
    #                             encoding="utf-8"
    #                             # engine="pyarrow"
    #                             )

    

    qdrant = QdrantWorking(client = QdrantClient(url=f"http://localhost:{PORT}", timeout=1200), 
                           collection = COLLECTION)

    results = []
    
    all_test_data = []

    for i in range(10):
        path_to_csv = rf"C:\Users\rudkevich-k\Desktop\patent_diploma\dataset_csv\dataset_for_test_fips_{i+1}.csv"
        df = pd.read_csv(path_to_csv, index_col=0)
        
        
        all_test_data.append(df)

    test_df = pd.concat(all_test_data, ignore_index=False)
    total_samples = len(test_df)
    print(f"Всего тестовых сэмплов: {total_samples}")


    


    for mode in mode_search:

        
        try:
            index = faiss.read_index(rf"C:\Users\rudkevich-k\Desktop\patent_diploma\pre_compute_embeddings_for_search\indexs\bge_{mode}_index.bin")
            

            print("Извлечение векторов из FAISS...")
            # reconstruct_n работает быстрее, чем цикл reconstruct
            test_vectors = np.zeros((total_samples, 1024), dtype=np.float32)
            # Внимание: это сработает ТОЛЬКО если ID в FAISS идут подряд 0..N-1 и совпадают с индексом в test_df
            try:
                index.reconstruct_n(0, total_samples, test_vectors)
            except Exception as e:
                print(f"Не удалось массово восстановить векторы: {e}")
                # Фолбэк на медленный метод, если нужно
                for idx in tqdm(range(total_samples)):
                    test_vectors[idx] = index.reconstruct(idx)

        except Exception as e:

            print(f"Ошибка загрузки индекса {mode}, пропускаем")
            print(e)

            continue
        
        start_index = 0
        results = []  # Сбрасываем перед каждым mode
        checkpoint_file = f"{mode}_results_of_{args.experiment}_cache_bge.pkl"

        if os.path.exists(checkpoint_file):
            print(f"📂 Найден чекпоинт: {checkpoint_file}. Восстановление...")
            try:
                with open(checkpoint_file, 'rb') as f:
                    saved_data = pickle.load(f)
                    results = saved_data['results']      # 🔑 ИСПРАВЛЕНО: присваиваем results
                    start_index = saved_data['last_index'] + 1
                    print(f"✅ Загружено {len(results)} записей. Продолжаем с индекса {start_index}")
            except Exception as e:
                print(f"⚠️ Ошибка чтения кэша: {e}. Начинаем с нуля.")
                start_index = 0
                results = []
        else:
            print("🆕 Кэш не найден. Начинаем поиск с нуля.")
            start_index = 0
            results = []



        print(f'Mode search: {mode}')

        for i in tqdm(range(start_index, total_samples), initial=start_index, total=total_samples, desc="Прогресс поиска"):

           
            cit_list = parse_patent_citations(test_df.loc[i, 'citations'])
            
            searched_docs = qdrant.search_docs(transform_query_vector, query_vec=test_vectors[i])
            # searched_docs = qdrant.search_docs(query_text)

            searched_docs_filtered = filter_query_doc(searched_docs, test_df.loc[i, "publication_number"]) # удалим id искомого документа из выдачи

            searched_docs_finally = [p.get('id') for p in searched_docs_filtered] # достаём только id

            gt = extract_gt(qdrant.collection, qdrant.client, cit_list)


            result = {}

            for k in [5, 10, 20, 30]:
                result[f'recall@{k}'] = calculate_recall_at_k(searched_docs_finally, gt, k)
                result[f'precision@{k}'] = calculate_precision_at_k(searched_docs_finally, gt, k)
                result[f'average_precision@{k}'] = calculate_average_precision(searched_docs_finally, gt, k)
                result[f'mrr@{k}'] = calculate_mrr(searched_docs_finally, gt, k)

            results.append(result)

            if (i + 1) % 5000 == 0:
                save_checkpoint(checkpoint_file, results, i)
        
        
        
        df_result = pd.DataFrame(results)


        df_result.to_csv(rf"C:\Users\rudkevich-k\Desktop\patent_diploma\tests_searching\result_bge\result_bge_{mode}_{args.experiment}.csv")


if __name__ == "__main__":
     main()
    


