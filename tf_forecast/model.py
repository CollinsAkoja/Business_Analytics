from tensorflow import keras
from tensorflow.keras import layers

def build_lstm_forecast(input_len=30):
    inp = keras.Input(shape=(input_len,1))
    x = layers.LSTM(128, return_sequences=False)(inp)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(64, activation='relu')(x)
    out = layers.Dense(1)(x)
    model = keras.Model(inp, out)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


def build_cnn_forecast(input_len=30):
    inp = keras.Input(shape=(input_len,1))
    x = layers.Conv1D(64, kernel_size=3, activation='relu')(inp)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Conv1D(128, kernel_size=3, activation='relu')(x)
    x = layers.GlobalMaxPooling1D()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(64, activation='relu')(x)
    out = layers.Dense(1)(x)
    model = keras.Model(inp, out)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model