# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping

# Adatok beolvasása
red_wine = pd.read_csv('https://raw.githubusercontent.com/karsarobert/Deep-Learning-2023/main/red-wine.csv')

# Az osztályok one-hot encodingja
Y = red_wine['quality']  # Osztályok
X = red_wine.drop(['quality'], axis=1)

# Tanuló- és teszthalmaz létrehozása
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1)

# Min-max skálázás
max_ = X_train.max(axis=0)
min_ = X_train.min(axis=0)
X_train = (X_train - min_) / (max_ - min_)
X_test = (X_test - min_) / (max_ - min_)

# One-hot encoding
Y_train = tf.keras.utils.to_categorical(Y_train, 10)
Y_test = tf.keras.utils.to_categorical(Y_test, 10)

# Korai megállás beállítása
early_stopping = EarlyStopping(
    min_delta=0.000000001,
    patience=50,
    restore_best_weights=True,
)

# Modell létrehozása
model = keras.Sequential([
    layers.Dense(1024, activation='relu', input_shape=(11,), name='input_layer'),
    layers.BatchNormalization(),
    layers.Dense(1024, activation='relu', input_shape=(11,), name='1_hidden_layer'),
    layers.BatchNormalization(),
    layers.Dense(1024, activation='relu', input_shape=(11,), name='2_hidden_layer'),
    layers.BatchNormalization(),
    layers.Dense(1024, activation='relu', input_shape=(11,), name='3_hidden_layer'),
    layers.BatchNormalization(),
    layers.Dropout(rate=0.3),
    layers.Dense(10, activation='exponential', input_shape=(11,), name='output_layer')  # Az osztályok száma 10, ahogy a one-hot encodingban is
])

model.compile(
    optimizer='sgd',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Modell tanítása
history = model.fit(
    X_train, Y_train,
    validation_data=(X_test, Y_test),
    batch_size=64,
    epochs=200,
    callbacks=[early_stopping]
)

# A tanulási folyamat vizualizálása
import matplotlib.pyplot as plt

history_df = pd.DataFrame(history.history)
history_df.loc[:, ['loss','val_loss','accuracy', 'val_accuracy']].plot()
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()
