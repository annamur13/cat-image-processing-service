import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from Utils import CSVConfig, PlotUtils


class WeatherAnalysis:
    def __init__(self, data_processor):
        self.data_processor = data_processor
        self.column_names = CSVConfig.COLUMN_NAMES
        PlotUtils.setup_plotting()

    def _process_temperature_data(self, file_path):
        location_data = {}
        state_monthly_data = {}

        reader = self.data_processor.read_csv_generator(file_path)
        cleaner = self.data_processor.clean_data_generator(reader)
        extractor = self.data_processor.extract_year_month_generator(cleaner)

        for chunk in extractor:
            for _, row in chunk.iterrows():
                location = row[self.column_names['location']]
                year = row['Year']
                temp = row[self.column_names['temperature']]

                if pd.notna(location) and pd.notna(year) and pd.notna(temp):
                    key = (location, year)
                    if key not in location_data:
                        location_data[key] = {'sum': 0, 'count': 0}
                    location_data[key]['sum'] += temp
                    location_data[key]['count'] += 1

                state = row[self.column_names['state']]
                month = row['Month']

                if pd.notna(state) and pd.notna(month) and pd.notna(temp):
                    if state not in state_monthly_data:
                        state_monthly_data[state] = {}
                    if month not in state_monthly_data[state]:
                        state_monthly_data[state][month] = []
                    state_monthly_data[state][month].append(temp)

        return location_data, state_monthly_data

    def task1_analysis(self, file_path):
        print("Сбор данных для задачи 1...")
        location_data, _ = self._process_temperature_data(file_path)

        if not location_data:
            raise ValueError("Не найдено данных о температурах по локациям")

        location_avg_temp = {}
        for (location, year), values in location_data.items():
            if location not in location_avg_temp:
                location_avg_temp[location] = []
            avg_temp = values['sum'] / values['count']
            location_avg_temp[location].append(avg_temp)

        result = {}
        for location, temps in location_avg_temp.items():
            if len(temps) >= 1:  # Хотя бы один год данных
                result[location] = np.mean(temps)

        if not result:
            raise ValueError("Недостаточно данных для анализа локаций")

        sorted_locations = sorted(result.items(), key=lambda x: x[1], reverse=True)
        top_3_hot = sorted_locations[:3]
        top_3_cold = sorted_locations[-3:] if len(sorted_locations) >= 6 else []

        print(f"Найдено {len(result)} локаций")
        return top_3_hot, top_3_cold

    def task2_analysis(self, file_path):
        print("Сбор данных для задачи 2...")
        _, state_monthly_data = self._process_temperature_data(file_path)

        if not state_monthly_data:
            raise ValueError("Не найдено данных о температурах по штатам")

        state_variance = {}
        for state, monthly_data in state_monthly_data.items():
            monthly_means = []
            for month in range(1, 13):
                if month in monthly_data and len(monthly_data[month]) > 0:
                    monthly_means.append(np.mean(monthly_data[month]))

            if len(monthly_means) >= 12:
                state_variance[state] = np.var(monthly_means)

        if not state_variance:
            raise ValueError("Не найдено штатов с полными данными за все месяцы")

        sorted_states = sorted(state_variance.items(), key=lambda x: x[1], reverse=True)
        top_3_high_var = sorted_states[:3]
        top_3_low_var = sorted_states[-3:] if len(sorted_states) >= 6 else []

        print(f"Найдено {len(state_variance)} штатов с полными данными")
        return top_3_high_var, top_3_low_var

    def task3_analysis(self, file_path):
        print("Анализ скорости ветра по штатам...")
        state_wind_data = {}

        reader = self.data_processor.read_csv_generator(file_path)
        cleaner = self.data_processor.clean_data_generator(reader)
        extractor = self.data_processor.extract_year_month_generator(cleaner)

        for chunk in extractor:
            for _, row in chunk.iterrows():
                state = row[self.column_names['state']]
                date = row[self.column_names['date']]
                wind_speed = row[self.column_names['wind_speed']]

                if (pd.notna(state) and pd.notna(date) and
                        pd.notna(wind_speed) and wind_speed >= 0):

                    if state not in state_wind_data:
                        state_wind_data[state] = {'dates': [], 'wind_speeds': []}

                    state_wind_data[state]['dates'].append(date)
                    state_wind_data[state]['wind_speeds'].append(wind_speed)

        if not state_wind_data:
            raise ValueError("Не найдено данных о скорости ветра")

        state_avg_wind = {}
        for state, data in state_wind_data.items():
            if len(data['wind_speeds']) > 10:
                state_avg_wind[state] = np.mean(data['wind_speeds'])

        if not state_avg_wind:
            raise ValueError("Недостаточно данных о скорости ветра для анализа")

        windiest_state = max(state_avg_wind.items(), key=lambda x: x[1])
        windiest_state_name = windiest_state[0]
        wind_data = state_wind_data[windiest_state_name]

        df = pd.DataFrame({
            'date': wind_data['dates'],
            'wind_speed': wind_data['wind_speeds']
        })
        df = df.sort_values('date').set_index('date')
        moving_avg = df['wind_speed'].rolling(
            window=CSVConfig.MOVING_AVERAGE_WINDOW,
            center=True,
            min_periods=1
        ).mean()

        print(f"Самый ветренный штат: {windiest_state_name} (ср. скорость: {windiest_state[1]:.2f} м/с)")
        return df['wind_speed'], moving_avg, windiest_state_name

    def additional_task_analysis(self, file_path):
        print("Анализ корреляции ветер-осадки...")
        wind_speeds = []
        precipitations = []

        reader = self.data_processor.read_csv_generator(file_path)
        cleaner = self.data_processor.clean_data_generator(reader)

        for chunk in cleaner:
            mask = (chunk[self.column_names['wind_speed']].notna() &
                    chunk[self.column_names['precipitation']].notna() &
                    (chunk[self.column_names['wind_speed']] >= 0) &
                    (chunk[self.column_names['precipitation']] >= 0))

            valid_data = chunk[mask]
            wind_speeds.extend(valid_data[self.column_names['wind_speed']].values)
            precipitations.extend(valid_data[self.column_names['precipitation']].values)

        if len(wind_speeds) < 2:
            raise ValueError("Недостаточно данных для корреляционного анализа")

        correlation = np.corrcoef(wind_speeds, precipitations)[0, 1]

        print(f"Проанализировано {len(wind_speeds)} пар данных")
        return np.array(wind_speeds), np.array(precipitations), correlation

    def plot_task1(self, top_3_hot, top_3_cold):
        if not top_3_hot and not top_3_cold:
            print("Нет данных для построения графика задачи 1")
            return

        locations = []
        temperatures = []
        colors = []
        labels = []

        if top_3_hot:
            locations.extend([loc[0] for loc in top_3_hot])
            temperatures.extend([temp[1] for temp in top_3_hot])
            colors.extend(['red'] * len(top_3_hot))
            labels.extend(['Самая высокая темп.'] * len(top_3_hot))

        if top_3_cold:
            locations.extend([loc[0] for loc in top_3_cold])
            temperatures.extend([temp[1] for temp in top_3_cold])
            colors.extend(['blue'] * len(top_3_cold))
            labels.extend(['Самая низкая темп.'] * len(top_3_cold))

        plt.figure(figsize=(12, 6))
        bars = plt.bar(locations, temperatures, color=colors, alpha=0.7)

        plt.title('Локации с самой высокой и самой низкой среднегодовой температурой', fontsize=14)
        plt.xlabel('Локации')
        plt.ylabel('Средняя температура (°C)')
        plt.xticks(rotation=45)
        plt.grid(axis='y', alpha=0.3)

        for bar, temp in zip(bars, temperatures):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                     f'{temp:.1f}°C', ha='center', va='bottom', fontweight='bold')

        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='red', alpha=0.7, label='Высокая температура'),
            Patch(facecolor='blue', alpha=0.7, label='Низкая температура')
        ]
        plt.legend(handles=legend_elements)

        PlotUtils.save_plot('task1_temperatures.png')
        PlotUtils.show_plot()

    def plot_task2(self, top_3_high_var, top_3_low_var):
        if not top_3_high_var and not top_3_low_var:
            print("Нет данных для построения графика задачи 2")
            return

        states = []
        variances = []
        colors = []

        if top_3_high_var:
            states.extend([state[0] for state in top_3_high_var])
            variances.extend([var[1] for var in top_3_high_var])
            colors.extend(['red'] * len(top_3_high_var))

        if top_3_low_var:
            states.extend([state[0] for state in top_3_low_var])
            variances.extend([var[1] for var in top_3_low_var])
            colors.extend(['blue'] * len(top_3_low_var))

        confidence_intervals = []
        for var in variances:
            n = 100
            df = n - 1
            chi2_low = stats.chi2.ppf(0.025, df)
            chi2_high = stats.chi2.ppf(0.975, df)
            ci_low = (df * var) / chi2_high
            ci_high = (df * var) / chi2_low
            confidence_intervals.append((ci_low, ci_high))

        plt.figure(figsize=(14, 8))
        x_pos = np.arange(len(states))

        bars = plt.bar(x_pos, variances, color=colors, alpha=0.7)

        for i, (bar, ci) in enumerate(zip(bars, confidence_intervals)):
            plt.errorbar(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                         yerr=[[bar.get_height() - ci[0]], [ci[1] - bar.get_height()]],
                         fmt='k', elinewidth=2, capsize=5, capthick=2)

        plt.title(
            'Штаты с самым высоким и низким разбросом среднемесячных температур\nс доверительными интервалами (95%)',
            fontsize=14)
        plt.xlabel('Штаты')
        plt.ylabel('Дисперсия температуры')
        plt.xticks(x_pos, states, rotation=45)
        plt.grid(axis='y', alpha=0.3)

        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='red', alpha=0.7, label='Высокий разброс'),
            Patch(facecolor='blue', alpha=0.7, label='Низкий разброс'),
            plt.Line2D([0], [0], color='black', linewidth=2, label='Доверительный интервал')
        ]
        plt.legend(handles=legend_elements)

        PlotUtils.save_plot('task2_variance.png')
        PlotUtils.show_plot()

    def plot_task3(self, original, moving_avg, state_name):
        plt.figure(figsize=(14, 8))

        if len(original) > 1000:
            sample_indices = np.random.choice(len(original), 1000, replace=False)
            sampled_dates = original.index[sample_indices]
            sampled_values = original.iloc[sample_indices]
            plt.plot(sampled_dates, sampled_values, 'b.', alpha=0.3,
                     markersize=1, label='Исходные данные (выборка)')
        else:
            plt.plot(original.index, original, 'b-', alpha=0.5,
                     linewidth=0.5, label='Исходные данные')

        plt.plot(moving_avg.index, moving_avg, 'r-', linewidth=2,
                 label=f'Скользящее среднее ({CSVConfig.MOVING_AVERAGE_WINDOW} дней)')

        plt.title(f'Скорость ветра в самом ветренном штате ({state_name})', fontsize=14)
        plt.xlabel('Дата')
        plt.ylabel('Скорость ветра (м/с)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        PlotUtils.save_plot('task3_wind_speed.png')
        PlotUtils.show_plot()

    def plot_additional_task(self, wind_speeds, precipitations, correlation):
        plt.figure(figsize=(12, 8))

        sample_size = min(5000, len(wind_speeds))
        if len(wind_speeds) > sample_size:
            indices = np.random.choice(len(wind_speeds), sample_size, replace=False)
        else:
            indices = np.arange(len(wind_speeds))

        plt.scatter(wind_speeds[indices], precipitations[indices],
                    alpha=0.6, s=2, color='purple', label='Данные измерений')

        plt.title(f'Корреляция между скоростью ветра и осадками\n(Коэффициент корреляции: {correlation:.3f})',
                  fontsize=14)
        plt.xlabel('Скорость ветра (м/с)')
        plt.ylabel('Осадки (мм)')
        plt.grid(True, alpha=0.3)

        if len(wind_speeds) > 1:
            z = np.polyfit(wind_speeds[indices], precipitations[indices], 1)
            p = np.poly1d(z)
            x_trend = np.linspace(wind_speeds[indices].min(), wind_speeds[indices].max(), 100)
            plt.plot(x_trend, p(x_trend), "r-", alpha=0.8, linewidth=2,
                     label='Линия тренда')

        plt.legend()
        PlotUtils.save_plot('additional_correlation.png')
        PlotUtils.show_plot()

    def print_results(self, top_3_hot, top_3_cold, top_3_high_var,
                      top_3_low_var, state_name, correlation):
        print("\n" + "=" * 60)
        print("РЕЗУЛЬТАТЫ АНАЛИЗА ПОГОДНЫХ ДАННЫХ")
        print("=" * 60)

        print("\nТОП-3 САМЫХ ГОРЯЧИХ ЛОКАЦИЙ:")
        for i, (loc, temp) in enumerate(top_3_hot, 1):
            print(f"   {i}. {loc}: {temp:.2f}°C")

        if top_3_cold:
            print("\nТОП-3 САМЫХ ХОЛОДНЫХ ЛОКАЦИЙ:")
            for i, (loc, temp) in enumerate(top_3_cold, 1):
                print(f"   {i}. {loc}: {temp:.2f}°C")

        print("\nТОП-3 ШТАТОВ С САМЫМ ВЫСОКИМ РАЗБРОСОМ ТЕМПЕРАТУР:")
        for i, (state, var) in enumerate(top_3_high_var, 1):
            print(f"   {i}. {state}: дисперсия = {var:.4f}")

        if top_3_low_var:
            print("\nТОП-3 ШТАТОВ С САМЫМ НИЗКИМ РАЗБРОСОМ ТЕМПЕРАТУР:")
            for i, (state, var) in enumerate(top_3_low_var, 1):
                print(f"   {i}.  {state}: дисперсия = {var:.4f}")

        print(f"\nСАМЫЙ ВЕТРЕННЫЙ ШТАТ: {state_name}")

        print(f"\nКОРРЕЛЯЦИЯ МЕЖДУ СКОРОСТЬЮ ВЕТРА И ОСАДКАМИ: {correlation:.3f}")

        if abs(correlation) < 0.1:
            strength = "очень слабая"
        elif abs(correlation) < 0.3:
            strength = "слабая"
        elif abs(correlation) < 0.5:
            strength = "умеренная"
        elif abs(correlation) < 0.7:
            strength = "заметная"
        else:
            strength = "сильная"

        direction = "положительная" if correlation > 0 else "отрицательная"
        print(f"   Характер связи: {strength} {direction}")

        print(f"\nГрафики сохранены в папке: '{CSVConfig.PLOTS_DIR}'")