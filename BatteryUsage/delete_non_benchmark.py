import pandas as pd

# Список файлов
files = ['RedmiNote9/power_log_cleaned', 'RedmiNote10ProBlue/power_log_cleaned', 'RedmiNote10ProPink/power_log_cleaned']

# Обработка каждого файла
for file in files:
    try:
        # Чтение CSV-файла
        df = pd.read_csv(file + ".csv")
        
        df = df[((df['brightness_pct'] == 0) & (df['cpu_load_pct'] == 0))]

        df.loc[df['display_brightness_lin'] == 0, 'display_brightness_lin'] = 0.00004

        
        # Сохранение обновленного файла
        df.to_csv(file + "_NO.csv", index=False)
        print(f"Файл {file} успешно обновлен.")
    except FileNotFoundError:
        print(f"Файл {file} не найден.")
    except Exception as e:
        print(f"Ошибка при обработке файла {file}: {e}")

