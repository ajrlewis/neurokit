from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np


class DataFrameProcessor:

    ScalingParams = Dict[str, Dict[str, float]]  # e.g. {"foo": {"min": 0.0, "max": 1.0}, ...}

    @staticmethod
    def add_temporal_columns(data: pd.DataFrame, column: str = "date") -> pd.DataFrame:
        week_scaling_param = 0.03773584905660377 * np.pi
        weekday_scaling_param = 0.2857142857142857 * np.pi
        isocalendar = pd.to_datetime(data[column]).dt.isocalendar()
        data["year"] = isocalendar.year
        data["week_sin"] = np.sin(week_scaling_param * isocalendar.week)
        data["week_cos"] = np.cos(week_scaling_param * isocalendar.week)
        data["weekday_sin"] = np.sin(weekday_scaling_param * isocalendar.day)
        data["weekday_cos"] = np.cos(weekday_scaling_param * isocalendar.day)
        data = data.drop(column, axis=1)
        return data

    @staticmethod
    def one_hot_encode_columns(data: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        data = pd.get_dummies(data, columns=columns, drop_first=True, dtype=int)
        return data

    @staticmethod
    def scale_columns(
        data: pd.DataFrame,
        columns: List[str],
        scaling_params: Optional[ScalingParams] = None,
    ) -> Tuple[pd.DataFrame, ScalingParams]:
        if scaling_params is None:
            scaling_params = {}
            for col in columns:
                min_value = data[col].min()
                max_value = data[col].max()
                scaling_params[col] = {"min": min_value, "max": max_value}
        else:
            for col in columns:
                if col not in scaling_params:
                    raise ValueError(
                        f"scaling_params does not contain scaling parameters for column '{col}'"
                    )
        for col in columns:
            min_value = scaling_params[col]["min"]
            max_value = scaling_params[col]["max"]
            data[col] = (data[col] - min_value) / (max_value - min_value)
        return data, scaling_params

    @staticmethod
    def extract(data: pd.DataFrame, columns: list) -> pd.DataFrame:
        return data.filter(regex="|".join([column for column in columns]))

    @staticmethod
    def split_data_by_date(data: pd.DataFrame, training_dataset_size: float = 0.8):
        unique_dates = data["date"].unique()
        split_index = int(len(unique_dates) * training_dataset_size)
        split_date = unique_dates[split_index]
        train_data = data[data["date"] < split_date]
        test_data = data[data["date"] >= split_date]
        return train_data, test_data
