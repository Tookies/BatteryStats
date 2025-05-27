import pandas as pd
import numpy as np

def clean_data(input_file, output_file):
    # Загружаем данные
    df = pd.read_csv(input_file)
    initial_count = len(df)
    print(f"Исходное количество записей: {initial_count}")

    # Конвертируем timestamp в формат datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    
    # Удаляем строки с некорректными timestamp
    df = df.dropna(subset=['timestamp'])
    records_timestamp_dropped = initial_count - len(df)
    print(f"Удалено записей с некорректным timestamp: {records_timestamp_dropped}")



    # Шаг 2: Определяем колонки, заканчивающиеся на '_ms', кроме 'sample_ms'
    ms_columns = [col for col in df.columns if col.endswith('_ms') and col != 'sample_ms']
    print(f"Найдено колонок _ms (кроме sample_ms): {len(ms_columns)}")

    # Шаг 3: Удаляем записи, где значения в ms_columns превышают sample_ms * 1.5
    records_ms_exceed_before = len(df)
    for col in ms_columns:
        df = df[df[col] <= df['sample_ms'] * 1.5]
    records_ms_exceed = records_ms_exceed_before - len(df)
    print(f"Удалено записей, где значения в _ms колонках превышают sample_ms * 1.5: {records_ms_exceed}")

    # Шаг 4: Сортируем по timestamp для корректного порядка
    df = df.sort_values(by='timestamp')

    # Шаг 5: Определяем запуски по разделителям (brightness_pct=50, cpu_load_pct=0)
    df['run_id'] = (df['brightness_pct'] == 0) & (df['cpu_load_pct'] == 0)
    print(f"Найдено разделителей (brightness_pct=0, cpu_load_pct=0): {df['run_id'].sum()}")
    df['run_id'] = df['run_id'].cumsum()  # Нумеруем запуски

    # Шаг 6: Удаляем первые две записи для каждой комбинации brightness_pct и cpu_load_pct в каждом запуске
    records_first_two_before = len(df)
    # Фильтруем записи, не являющиеся разделителями
    df_non_delimiters = df[(df['brightness_pct'] != 0) | (df['cpu_load_pct'] != 0)]
    print(f"Количество записей без разделителей: {len(df_non_delimiters)}")
    # Группируем по run_id, brightness_pct, cpu_load_pct и удаляем первые две записи
    df_cleaned = df_non_delimiters.groupby(['run_id', 'brightness_pct', 'cpu_load_pct'], group_keys=False).apply(
        lambda x: x.iloc[2:], include_groups=False
    ).reset_index(drop=True)
    
    # Добавляем обратно записи-разделители
    df_delimiters = df[(df['brightness_pct'] == 0) & (df['cpu_load_pct'] == 0)]
    print(f"Количество разделителей: {len(df_delimiters)}")
    if not df_delimiters.empty:
        df_cleaned = pd.concat([df_cleaned, df_delimiters]).sort_values(by='timestamp').reset_index(drop=True)
    else:
        df_cleaned = df_cleaned.sort_values(by='timestamp').reset_index(drop=True)

    records_first_two = records_first_two_before - len(df_cleaned)
    print(f"Удалено первых двух записей для каждой комбинации в каждом запуске: {records_first_two}")

    # Удаляем временный столбец run_id, если он остался
    if 'run_id' in df_cleaned.columns:
        df_cleaned = df_cleaned.drop(columns=['run_id'])

    # Итоговая статистика
    print(f"\nИтоговое количество записей после очистки: {len(df_cleaned)}")
    print(f"Всего удалено записей: {initial_count - len(df_cleaned)}")
    print(f"Детализация удалений:")
    print(f"- По некорректному timestamp: {records_timestamp_dropped}")
    print(f"- По критерию sample_ms * 1.5: {records_ms_exceed}")
    print(f"- Первые две записи для каждой комбинации в каждом запуске: {records_first_two}")

    # Сохраняем очищенные данные
    df_cleaned.to_csv(output_file, index=False)
    print(f"\nОчищенные данные сохранены в {output_file}")

# Пример использования
input_files = [
    'RedmiNote9/power_log.csv',
    'RedmiNote10ProBlue/power_log.csv',
    'RedmiNote10ProPink/power_log.csv'
]
output_files = [
    'RedmiNote9/power_log_cleaned.csv',
    'RedmiNote10ProBlue/power_log_cleaned.csv',
    'RedmiNote10ProPink/power_log_cleaned.csv'
]

for input_file, output_file in zip(input_files, output_files):
    clean_data(input_file, output_file)