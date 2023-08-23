import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import ModelCheckpoint


class LSTMModel:
    def __init__(self, features, target):
        self.features = features
        self.target = target
        self.model = None

    def create_model(self):
        self.model = Sequential()
        self.model.add(LSTM(units=64, input_shape=(1, self.features.shape[1])))
        self.model.add(Dense(units=32, activation="relu"))
        self.model.add(Dense(units=1))
        self.model.compile(optimizer="adam", loss="mean_squared_error")

    def train_model(self, epochs=10, batch_size=32):
        X_train, X_val, y_train, y_val = train_test_split(
            self.features, self.target, test_size=0.2, shuffle=False
        )
        X_train_reshaped = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
        X_val_reshaped = np.reshape(X_val, (X_val.shape[0], 1, X_val.shape[1]))
        checkpoint = ModelCheckpoint(
            "best_model.h5", save_best_only=True, save_weights_only=False
        )
        self.model.fit(
            X_train_reshaped,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val_reshaped, y_val),
            callbacks=[checkpoint],
        )

    def save_model(self, filename="trained_model.h5"):
        self.model.save(filename)

    def forecast(self, data):
        data_reshaped = np.reshape(data, (data.shape[0], 1, data.shape[1]))
        predictions = self.model.predict(data_reshaped)
        return predictions


def main():
    lstm_model = LSTMModel(features, target)
    lstm_model.create_model()
    lstm_model.train_model(epochs=10, batch_size=32)
    lstm_model.save_model()
    predictions = lstm_model.forecast(new_data)


if __name__ == "__main__":
    main()
