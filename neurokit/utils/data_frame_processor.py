from typing import List
import pandas as pd
import numpy as np


class DataFrameProcessor:
    @staticmethod
    def add_temporal_columns(data: pd.DataFrame, column: str = "date") -> pd.DataFrame:
        isocalendar = pd.to_datetime(data[column]).dt.isocalendar()
        data["year"] = isocalendar.year
        data["week_sin"] = np.sin(2 * np.pi * isocalendar.week / 53)
        data["week_cos"] = np.cos(2 * np.pi * isocalendar.week / 53)
        data["weekday_sin"] = np.sin(2 * np.pi * isocalendar.day / 7)
        data["weekday_cos"] = np.cos(2 * np.pi * isocalendar.day / 7)
        data = data.drop(column, axis=1)
        return data

    @staticmethod
    def one_hot_encode_columns(data: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        data = pd.get_dummies(data, columns=columns, drop_first=True, dtype=int)
        return data

    @staticmethod
    def scale_columns(data: pd.DataFrame, scaling_params=None) -> pd.DataFrame:
        scaler = MinMaxScaler()
        scaled_data = data.copy()

        if scaling_params is None:
            scaling_params = {}

        for column in scaled_data.columns:
            if column not in scaling_params:
                scaling_params[column] = {
                    "min": scaled_data[column].min(),
                    "max": scaled_data[column].max(),
                }

            scaled_data[column] = scaler.fit_transform(
                scaled_data[column].values.reshape(-1, 1)
            )

        return scaled_data, scaling_params

    @staticmethod
    def split_data_by_date(data: pd.DataFrame, training_dataset_size: float = 0.8):
        """
        Split the data DataFrame into training and test samples based on the date column.

        Parameters:
        - data (pd.DataFrame): The input DataFrame to be split.
        - training_dataset_size (float): The percentage of data to be used for training.

        Returns:
        - train_data (pd.DataFrame): The training dataset.
        - test_data (pd.DataFrame): The test dataset.
        """

        # Get the unique dates in the dataset
        unique_dates = data["date"].unique()

        # Calculate the index to split the data based on the training dataset size
        split_index = int(len(unique_dates) * training_dataset_size)

        # Get the date at the split index
        split_date = unique_dates[split_index]

        # Split the data into training and test sets based on the split date
        train_data = data[data["date"] < split_date]
        test_data = data[data["date"] >= split_date]

        return train_data, test_data
