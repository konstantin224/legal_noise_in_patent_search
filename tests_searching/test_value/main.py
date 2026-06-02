import pandas as pd
import numpy as np



import os

from scipy import stats

def compare_search_results(df_a: pd.DataFrame, df_b: pd.DataFrame, name) -> pd.DataFrame:
    """
    Сравнивает два датафрейма с метриками поиска.
    
    Args:
        df_a: Результаты первого метода (Baseline).
        df_b: Результаты второго метода (Experiment/Delta).
        id_col: Название колонки с ID запроса.
        
    Returns:
        DataFrame со сводной статистикой по каждой метрике.
    """

    id_col = 'query_id'

    ids_a = set(df_a[id_col])
    ids_b = set(df_b[id_col])
    common_ids = ids_a & ids_b
    
    if len(common_ids) != len(ids_a) or len(common_ids) != len(ids_b):
        print(f"⚠️ Внимание: Не все ID совпадают. Общие: {len(common_ids)}, Только в A: {len(ids_a - ids_b)}, Только в B: {len(ids_b - ids_a)}")
        # Фильтруем только общие, чтобы сравнение было корректным
        df_a = df_a[df_a[id_col].isin(common_ids)].reset_index(drop=True)
        df_b = df_b[df_b[id_col].isin(common_ids)].reset_index(drop=True)

    summary_stats_p_value = []
    summary_stats_sigificant = []

    metrics = df_a.columns
    
    for metric in metrics:

        if metric == "query_id":
            continue
        t_stat, p_value = stats.ttest_rel(df_a[metric].dropna(), df_b[metric].dropna())
        
        summary_stats_p_value.append({
            'Metric': metric,
            'name_exp': name,
            "p-value": p_value
        })

        summary_stats_sigificant.append({
            'Metric': metric,
            'name_exp': name,
            'Significant': '✅ Yes' if p_value < 0.05 else '❌ No'
        })

        
    summary_df_pvalue = pd.DataFrame(summary_stats_p_value)
    summary_df_significat = pd.DataFrame(summary_stats_sigificant)
    
    return summary_df_pvalue, summary_df_significat

def main():

    name_mode = "description"

    name_test = f"result_{name_mode}"

    df_pvalue = pd.DataFrame(columns = ['Metric','name_exp', 'p-value'])

    name_baseline = f"{name_test}_baseline.csv"

    df_significat = pd.DataFrame(columns = ['Metric', 'name_exp', 'Significant'])

    path = r"C:\Users\rudkevich-k\Desktop\patent_diploma\tests_searching\result"

    df_base = pd.read_csv(rf"{path}\{name_baseline}", index_col = 0)

    df_base['query_id'] = df_base.index
    

    for name in os.listdir(path):

        if name == name_baseline:
            continue

        if name.startswith(name_test):

            df_test = pd.read_csv(rf"{path}\{name}", index_col = 0)
            df_test['query_id'] = df_test.index

            new_df_pvalue, new_df_significat = compare_search_results(df_base, df_test, name)

            df_pvalue = pd.concat([df_pvalue, new_df_pvalue], axis = 0, ignore_index=True)

            df_significat = pd.concat([df_significat, new_df_significat], axis = 0, ignore_index=True)

    print("goyda")
    df_pvalue.to_csv(rf"C:\Users\rudkevich-k\Desktop\patent_diploma\tests_searching\test_value\p_value_test_{name_test}.csv")
    df_significat.to_csv(rf"C:\Users\rudkevich-k\Desktop\patent_diploma\tests_searching\test_value\significat_test_{name_test}.csv")

if __name__ == "__main__":
     main()



