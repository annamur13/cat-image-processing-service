from abc import ABC, abstractmethod


class IDataProcessing(ABC):
    @abstractmethod
    def read_csv_generator(self, file_path):
        pass

    @abstractmethod
    def clean_data_generator(self, data_generator):
        pass

    @abstractmethod
    def extract_year_month_generator(self, data_generator):
        pass