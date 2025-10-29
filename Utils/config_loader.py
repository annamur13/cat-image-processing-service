import os

class CSVConfig:
    DATA_FILE = 'weather.csv'
    PLOTS_DIR = 'plots'
    CHUNKSIZE = 10000
    MOVING_AVERAGE_WINDOW = 30

    COLUMN_NAMES = {
        'location': 'Station.City',
        'state': 'Station.State',
        'date': 'Date.Full',
        'temperature': 'Data.Temperature.Avg Temp',
        'wind_speed': 'Data.Wind.Speed',
        'precipitation': 'Data.Precipitation'
    }

    @classmethod
    def setup_directories(cls):
        if not os.path.exists(cls.PLOTS_DIR):
            os.makedirs(cls.PLOTS_DIR)

    @classmethod
    def get_data_columns(cls):
        return [
            cls.COLUMN_NAMES['location'],
            cls.COLUMN_NAMES['state'],
            cls.COLUMN_NAMES['date'],
            cls.COLUMN_NAMES['temperature'],
            cls.COLUMN_NAMES['wind_speed'],
            cls.COLUMN_NAMES['precipitation']
        ]

    @classmethod
    def print_column_mapping(cls):
        print("Соответствие колонок в конфиге:")
        for key, value in cls.COLUMN_NAMES.items():
            print(f"   {key:15} → {value}")