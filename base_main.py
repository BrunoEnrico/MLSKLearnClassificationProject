from data_processing import DataProcessing


class Main:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.data = self.load_data()

    def load_data(self):
        dp = DataProcessing()
        data = dp.get_csv(self.filepath)
        return data
