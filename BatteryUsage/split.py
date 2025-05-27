from sklearn.model_selection import train_test_split
import pandas as pd

path = 'RedmiNote10ProBlue/'
# Загружаем данные
df = pd.read_csv(path + 'power_log_cleaned_.csv')  # Твой файл для RedmiNote10ProPink

# Разделение: 70% train, 30% temp
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Сохранение в файлы
train_df.to_csv(path + 'train.csv', index=False)
test_df.to_csv(path + 'test.csv', index=False)

print(f"Train: {len(train_df)} samples, Test: {len(test_df)} samples")