import os
import pandas as pd
from implementation.data_processing import DataProcessing
from implementation.weather_analysis import WeatherAnalysis
from Utils import CSVConfig


def main():

    print("ЗАПУСК АНАЛИЗА ПОГОДНЫХ ДАННЫХ")
    print("=" * 50)

    CSVConfig.setup_directories()

    CSVConfig.print_column_mapping()

    if not os.path.exists(CSVConfig.DATA_FILE):
        print(f"Файл {CSVConfig.DATA_FILE} не найден!")
        print("Текущая директория:", os.getcwd())
        print("Содержимое текущей директории:")
        for file in os.listdir('.'):
            print(f"   - {file}")
        return

    print(f"Файл {CSVConfig.DATA_FILE} найден!")

    try:
        data_processor = DataProcessing()
        weather_analyzer = WeatherAnalysis(data_processor)

        print("\nПРОВЕРКА ДАННЫХ...")

        reader = data_processor.read_csv_generator(CSVConfig.DATA_FILE)
        first_chunk = next(reader)

        print(f"Размер чанка: {len(first_chunk)} строк")
        print(f"Колонки: {list(first_chunk.columns)}")

        required_columns = CSVConfig.get_data_columns()
        actual_columns = list(first_chunk.columns)

        missing_columns = [col for col in required_columns if col not in actual_columns]
        if missing_columns:
            print(f"Отсутствуют колонки: {missing_columns}")
            return
        else:
            print("Все необходимые колонки присутствуют")

        print("\nВЫПОЛНЕНИЕ АНАЛИЗА...")

        print("\n1️ВЫПОЛНЕНИЕ ЗАДАЧИ 1...")
        try:
            top_3_hot, top_3_cold = weather_analyzer.task1_analysis(CSVConfig.DATA_FILE)
            print("Задача 1 завершена")
        except Exception as e:
            print(f"Ошибка в задаче 1: {e}")
            top_3_hot, top_3_cold = [], []

        print("\n ВЫПОЛНЕНИЕ ЗАДАЧИ 2...")
        try:
            top_3_high_var, top_3_low_var = weather_analyzer.task2_analysis(CSVConfig.DATA_FILE)
            print("Задача 2 завершена")
        except Exception as e:
            print(f"Ошибка в задаче 2: {e}")
            top_3_high_var, top_3_low_var = [], []

        print("\n ВЫПОЛНЕНИЕ ЗАДАЧИ 3...")
        try:
            wind_original, wind_moving_avg, windiest_state = weather_analyzer.task3_analysis(CSVConfig.DATA_FILE)
            print("Задача 3 завершена")
        except Exception as e:
            print(f"Ошибка в задаче 3: {e}")
            wind_original, wind_moving_avg, windiest_state = None, None, "Не определен"

        print("\nВЫПОЛНЕНИЕ ДОПОЛНИТЕЛЬНОЙ ЗАДАЧИ...")
        try:
            wind_speeds, precipitations, correlation = weather_analyzer.additional_task_analysis(CSVConfig.DATA_FILE)
            print("Дополнительная задача завершена")
        except Exception as e:
            print(f"Ошибка в дополнительной задаче: {e}")
            wind_speeds, precipitations, correlation = None, None, 0

        print("\nПОСТРОЕНИЕ ГРАФИКОВ...")

        if top_3_hot or top_3_cold:
            print("График для задачи 1...")
            weather_analyzer.plot_task1(top_3_hot, top_3_cold)
        else:
            print("Нет данных для графика задачи 1")

        if top_3_high_var or top_3_low_var:
            print("График для задачи 2...")
            weather_analyzer.plot_task2(top_3_high_var, top_3_low_var)
        else:
            print("Нет данных для графика задачи 2")

        if wind_original is not None and wind_moving_avg is not None:
            print("График для задачи 3...")
            weather_analyzer.plot_task3(wind_original, wind_moving_avg, windiest_state)
        else:
            print("Нет данных для графика задачи 3")

        if wind_speeds is not None and precipitations is not None:
            print("График для дополнительной задачи...")
            weather_analyzer.plot_additional_task(wind_speeds, precipitations, correlation)
        else:
            print("Нет данных для графика дополнительной задачи")

        print("\nВЫВОД РЕЗУЛЬТАТОВ...")
        weather_analyzer.print_results(
            top_3_hot, top_3_cold,
            top_3_high_var, top_3_low_var,
            windiest_state, correlation
        )

        print(f"\nАНАЛИЗ ЗАВЕРШЕН! Графики сохранены в папке '{CSVConfig.PLOTS_DIR}'")

    except Exception as e:
        print(f"Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()