# implementation/data_processing.py
import pandas as pd
import numpy as np
from interfaces.i_data_processing import IDataProcessing
from Utils import CSVConfig


class DataProcessing(IDataProcessing):
    def __init__(self, chunksize=CSVConfig.CHUNKSIZE):
        self.chunksize = chunksize
        self.column_names = CSVConfig.COLUMN_NAMES

    def read_csv_generator(self, file_path):
        print(f"Чтение файла {file_path} с chunksize={self.chunksize}")
        for chunk in pd.read_csv(file_path, chunksize=self.chunksize, low_memory=False):
            print(f"Прочитано {len(chunk)} строк")
            yield chunk

    def clean_data_generator(self, data_generator):
        required_columns = CSVConfig.get_data_columns()

        for chunk_num, chunk in enumerate(data_generator):
            print(f"Очистка чанка {chunk_num + 1}...")

            existing_columns = [col for col in required_columns if col in chunk.columns]
            missing_columns = [col for col in required_columns if col not in chunk.columns]

            if missing_columns:
                print(f"Отсутствуют колонки: {missing_columns}")

            if not existing_columns:
                print("Нет нужных колонок для обработки, пропускаем чанк")
                continue

            original_count = len(chunk)

            chunk = chunk.dropna(subset=existing_columns)

            if self.column_names['date'] in chunk.columns:
                chunk[self.column_names['date']] = pd.to_datetime(
                    chunk[self.column_names['date']], errors='coerce'
                )
                chunk = chunk.dropna(subset=[self.column_names['date']])

            numeric_columns = [
                self.column_names['temperature'],
                self.column_names['wind_speed'],
                self.column_names['precipitation']
            ]

            for col in numeric_columns:
                if col in chunk.columns:
                    chunk[col] = pd.to_numeric(chunk[col], errors='coerce')
                    if col in [self.column_names['wind_speed'], self.column_names['precipitation']]:
                        chunk = chunk[chunk[col] >= 0]

            cleaned_count = len(chunk)
            print(f"Очищено: {original_count} → {cleaned_count} строк")

            yield chunk

    def extract_year_month_generator(self, data_generator):
        """Генератор для извлечения года и месяца"""
        for chunk_num, chunk in enumerate(data_generator):
            print(f"Извлечение дат из чанка {chunk_num + 1}...")

            if self.column_names['date'] in chunk.columns:
                chunk['Year'] = chunk[self.column_names['date']].dt.year
                chunk['Month'] = chunk[self.column_names['date']].dt.month
                print(f"Добавлены колонки Year и Month")
            else:
                print(f"Колонка даты не найдена")

            yield chunk