import pandas as pd
import glob

# Шаг 1: Получаем список всех CSV-файлов в папке
csv_files = glob.glob("results_*.csv")

# Шаг 2: Читаем каждый файл и добавляем в список DataFrames
dfs = []
for file in csv_files:
    df = pd.read_csv(file)
    # Добавляем колонку с именем файла для отслеживания источника
    df['source_file'] = file  
    dfs.append(df)

# Шаг 3: Объединяем все DataFrame в один
concatenated_df = pd.concat(dfs, ignore_index=True)

# Шаг 4: Сохраняем результат в новый CSV (опционально)
concatenated_df.to_csv("concatenated_results.csv", index=False)

# Выводим первые 5 строк для проверки
print(concatenated_df.head())