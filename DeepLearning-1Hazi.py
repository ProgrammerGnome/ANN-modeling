import tensorflow as tf
from tensorflow import keras

# Beállítások a tréninghez
EPOCHS = 5 # korszakok száma: 50
BATCH_SIZE = 128
VERBOSE = 1 # lássuk az edzés adatait
NB_CLASSES = 10   # kimenetek száma
#N_HIDDEN = 128

# MNIST dataset letöltése
# verify
# the split between train and test is 60,000, and 10,000 respectly
mnist = keras.datasets.mnist
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

#print(X_train[0])

# X_train 60000 adat 28x28-as felbontásban --> átalakítása 60000 x 784 alakzatra
RESHAPED = 784
#
X_train = X_train.reshape(60000, -1)
X_test = X_test.reshape(10000, RESHAPED)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

#print(X_train[0,:])

# Adatok normalizálása [0,1]
X_train /= 255
X_test /= 255
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

#print(X_train[0,:])
#print(Y_test[0])

# One-hot kódolás
Y_train = tf.keras.utils.to_categorical(Y_train, 10)
Y_test = tf.keras.utils.to_categorical(Y_test, 10)

#print(Y_test[0])

# Model felépítése
model = tf.keras.models.Sequential()
model.add(keras.layers.Dense(10, input_shape=(784,), name='hidden_layer', activation='relu'))
model.add(keras.layers.Dense(100, input_shape=(784,), name='hidden_layer2', activation='sigmoid'))
model.add(keras.layers.Dense(10, input_shape=(784,), name='dense_layer', activation='softmax'))

# A modell paramétereinek kiíratása
model.summary()

# A modell lefordítása
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
              
# A modell képzése
history = model.fit(X_train, Y_train,
		batch_size=128, epochs=5,
		verbose=1, validation_data=(X_test, Y_test))

import pandas as pd

# Convert the training history to a dataframe
history_df = pd.DataFrame(history.history)
#history_df.loc[:, ['loss','accuracy','val_loss', 'val_accuracy']].plot();
history_df.loc[:, ['loss','val_loss','accuracy', 'val_accuracy']].plot();

import numpy as np

# Tesztadatokon való predikció
Y_pred = model.predict(X_test)

# A predikciókat visszaváltjuk osztályokra (a legvalószínűbb osztály)
Y_pred_classes = np.argmax(Y_pred, axis=1)

# Az eredeti osztályokat is visszaállítjuk one-hot kódolásból
Y_true = np.argmax(Y_test, axis=1)

# A predikciók és az eredeti osztályok kiíratása
for i in range(10):  # Itt az első 10 tesztesetet mutatjuk be
    print(f"Eredeti osztály: {Y_true[i]}, Predikált osztály: {Y_pred_classes[i]}")
