from datetime import datetime

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.axes import Axes


class DataProcessing:
    def __init__(self):
        pass

    @staticmethod
    def get_csv(path: str) -> pd.DataFrame:
        return pd.read_csv(path)

    @staticmethod
    def map_column(data: pd.Series, column_to_map: dict) -> pd.Series:
        return data.map(column_to_map)

    @staticmethod
    def get_scatterplot(data: pd.DataFrame, x: str, y: str, **kwargs) -> Axes:
        return sns.scatterplot(data=data, x=x, y=y, **kwargs)

    @staticmethod
    def show_scatterplot() -> None:
        plt.show()

    @staticmethod
    def get_query(data: pd.DataFrame, query: str) -> pd.DataFrame:
        return data.query(query)

    @staticmethod
    def convert_miles_to_km(data: pd.DataFrame, column: str, new_column: str = "", decimal: int = 2) -> pd.DataFrame:
        data[new_column or column] = round(data[column] * 1.60934, decimal)
        return data

    @staticmethod
    def get_age(data: pd.DataFrame, column: str, new_column: str = "") -> None:
        data[new_column or column] = datetime.today().year - data[column]

    @staticmethod
    def drop_columns(data: pd.DataFrame, column: str | list[str]) -> None:
        data.drop(column, axis=1, inplace=True)
