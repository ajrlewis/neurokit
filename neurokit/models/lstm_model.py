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
    def __init__(self, features: np.array, target: np.array):
        self.features = features
        self.target = target
        self.model = None

    def create_model(self):
        self.model = Sequential()
        self.model.add(
            LSTM(
                units=64,
                input_shape=(self.features.shape[1], 1),
                return_sequences=False,
            )
        )
        self.model.add(Dense(units=32, activation="relu"))
        self.model.add(Dense(units=1))
        self.model.compile(optimizer="adam", loss="mean_absolute_error")

    def train_model(self, epochs: int = 50, batch_size: int = 128):
        X_train, X_val, y_train, y_val = train_test_split(
            self.features, self.target, test_size=0.2, shuffle=True
        )
        X_train_reshaped = reshape(X_train)
        X_val_reshaped = reshape(X_val)
        early_stopping = EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True
        )
        self.model.fit(
            X_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=[early_stopping],
        )

    def forecast(self, data: np.array) -> np.array:
        data_reshaped = reshape(data)
        predictions = self.model.predict(data_reshaped)
        return predictions.flatten()

    def save_model(self, filename: str = "model.keras"):
        self.model.save(filename)

    def load_model(self, filename: str = "model.keras"):
        self.model = load_model(filename)

    def model_architecture(self):
        return self.model.to_json()

    def model_weights(self):
        return self.model.get_weights()


def main():
    lstm_model = LSTMModel(features, target)
    lstm_model.create_model()
    lstm_model.train_model(epochs=10, batch_size=32)
    lstm_model.save_model()
    predictions = lstm_model.forecast(new_data)


if __name__ == "__main__":
    main()
