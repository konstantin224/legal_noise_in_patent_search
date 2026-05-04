import os
import sys
import torch
import pandas as pd

from tqdm import tqdm
from system_path import *
from work_with_database import *
from get_patent_from_xml import *
# from __future__ import annotations



os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

print("🐍 Интерпретатор:", sys.executable)
print("📦 Версия torch:", torch.__version__)
print("🎮 CUDA в сборке:", torch.version.cuda)
print("✅ GPU доступен:", torch.cuda.is_available())




list_name_file = []

for name in os.listdir(PATH_TO_ALL_FILES):
    for name_ in os.listdir(PATH_TO_ALL_FILES + f"/{name}"):
      list_name_file.append(f"{PATH_TO_ALL_FILES}/{name}/{name_}/document.xml") 



# Инициализируем датафреймы и процесс индексации базы данных
test_dataset = pd.DataFrame(columns = COLUMNS)
df_values = pd.DataFrame(columns = COLUMNS)

for name in tqdm(list_name_file, desc='Total'):

    try:
        df = build_df_from_st96(name)
    except Exception as e:
        print(f'Ошибка в компиляции: {name}')
        print(f'{e}')
    
    if df is None:
        continue

    df_values = pd.concat([df_values, df], axis = 0, ignore_index = True)

    if len(df_values) == BATCH_SIZE:
        test_dataset = index_docs_fast(df = df_values, test_dataset = test_dataset)
        df_values = pd.DataFrame(columns = COLUMNS)
        
# На случай, если останутся ещё семплы в выборке
test_dataset = index_docs_fast(df = df, test_dataset = test_dataset)
