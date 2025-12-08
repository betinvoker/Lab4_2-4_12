import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras import utils, models, layers
from keras.models import Sequential
from keras.layers import *

data = pd.read_csv('./4_8/bug_report.csv', sep=';')
data['Дата'] = pd.to_datetime(data['Дата'], format='%d.%m.%Y')

df = pd.DataFrame(data)

df_unique_tip = df['Тип ошибки'].unique()
print(df_unique_tip)

df_unique_pr = df['Приоритет'].unique()
print(df_unique_pr)

label_encoder_tip = LabelEncoder()
label_encoder_pr = LabelEncoder()
df['Тип ошибки'] = label_encoder_tip.fit_transform(df['Тип ошибки'])
df['Приоритет'] = label_encoder_pr.fit_transform(df['Приоритет'])
print(df['Тип ошибки'])

x = df['Тип ошибки'].values
y = df['Приоритет'].values

x = x.reshape(-1, 1, 1)

x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                                    test_size=0.2,
                                                    random_state=42)

y_train = utils.to_categorical(y_train, 4)
y_test = utils.to_categorical(y_test, 4)

print(y_train.shape)
print(y_train)

x_train_lstm = x_train.reshape(-1, 1, 1)  # 1 временной шаг, 1 признак
x_test_lstm = x_test.reshape(-1, 1, 1)

print(x_train_lstm.shape)
print(y_train)

model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape=(1, 1)))
model.add(Dropout(0.2))
model.add(LSTM(64, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(32, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(10))
model.add(Dropout(0.2))
model.add(Dense(4, activation="softmax"))

model.compile(optimizer='adam', loss='categorical_crossentropy', 
              metrics=['accuracy'],)

history1 = model.fit(x_train, y_train, epochs=50, batch_size=32)

plt.plot(history1.history['accuracy'], label='accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

loss, accuracy = model.evaluate(x_test, y_test)
print(f'Тестировать точность: {accuracy:.2f}')

pred = model.predict(x_test)

print(y_test[0])
print(pred[0])
print(max(pred[0]))

print(df_unique_tip)
x_unique = df['Тип ошибки'].unique()
print(x_unique)

print(df_unique_pr)
y_unique = df['Приоритет'].unique()
print(y_unique)
x_unique_reshaped = x_unique.reshape(-1, 1, 1)

x_test_pred = pd.DataFrame(x_unique, columns=['Тип ошибки'])
print(x_test_pred)

pred0 = model.predict(x_unique_reshaped)

for i in range(len(pred0)):
    error_type_text = label_encoder_tip.inverse_transform([x_unique[i]])[0]
    print(x_test_pred[i:i+1], x_unique[i:i+1], 'лог')
    print(pred0[i])
    print('max =', np.max(pred0[i]), '\tПриоритет =', np.argmax(pred0[i]))

model = Sequential()
model.add(Input(shape=(x_train.shape[0],1)))
model.add(GRU(32))
model.add(Dropout(0.2))
model.add(Dense(4, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])

history1 = model.fit(x_train, y_train, epochs=50, batch_size=32)

plt.plot(history1.history['accuracy'], label='accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()