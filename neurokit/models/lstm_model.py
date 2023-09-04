from __future__ import annotations
from typing import List, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from keras.layers import LSTM, Dense, Dropout
from keras.models import load_model, Sequential
from neurokit.utils.data_frame_processor import DataFrameProcessor


def reshape(data: np.array) -> np.array:
    return data.reshape(data.shape[0], data.shape[1], 1)


class LSTMModel:
    def __init__(
        self, feature_size: int, target_size: int, model: Optional[Sequential] = None
    ):
        self.batch_size = None  # Variable number of samples in a batch
        self.feature_size = feature_size
        self.target_size = target_size
        self._model = model
        if not self._model:
            self._create()

    def __repr__(self) -> str:
        return f"{self._model} with {self.feature_size} features and {self.target_size} targets"

    @property
    def architecture(self) -> str:
        return self._model.to_json()

    @property
    def weights(self) -> List[np.array]:
        return self._model.get_weights()

    def _create(self):
        self._model = Sequential()
        self._model.add(
            LSTM(
                units=64,
                input_shape=(self.feature_size, self.target_size),
                return_sequences=False,
            )
        )
        self._model.add(Dense(units=32, activation="relu"))
        self._model.add(Dense(units=1))
        self._model.compile(optimizer="adam", loss="mean_absolute_error")

    def train(
        self,
        features: np.array,
        targets: np.array,
        epochs: int = 50,
        batch_size: int = 128,
    ):
        X_train, X_val, y_train, y_val = train_test_split(
            features, targets, test_size=0.2, shuffle=True
        )
        X_train_reshaped = reshape(X_train)
        X_val_reshaped = reshape(X_val)
        early_stopping = EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True
        )
        self._model.fit(
            X_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=[early_stopping],
        )

    def predict(self, features: np.array) -> np.array:
        features_reshaped = reshape(features)
        predictions = self._model.predict(features_reshaped)
        return predictions.flatten()

    def save(self, filename: str = "model.keras"):
        self._model.save(filename)

    @classmethod
    def load(cls, filename: str = "model.keras") -> LSTMModel:
        model = load_model(filename)
        batch_size, feature_size, target_size = model.layers[0].input_shape
        lstm_model = cls(
            feature_size=feature_size, target_size=target_size, model=model
        )
        return lstm_model
