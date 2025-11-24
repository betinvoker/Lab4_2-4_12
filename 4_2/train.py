import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import ssl
import tensorflow as tf
from tensorflow import keras
from keras import layers

# Отключение проверки SSL (только для разработки)
ssl._create_default_https_context = ssl._create_unverified_context

from keras.datasets import cifar10
from keras.preprocessing import image
import matplotlib.pyplot as plt
from PIL import Image #Для отрисовки изображений
import numpy as np
import random #Для генерации случайных чисел 
from tensorflow.keras.callbacks import TensorBoard
import datetime

tf.random.set_seed(1)

log_dir = "./4_2/logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

#Определяем названия классов по порядку, их соответственно 10
classes = ['самолет', 'автомобиль', 'птица', 'кот', 'олень', 'собака',
'лягушка', 'лошадь', 'корабль', 'грузовик']

#Выводим для примера картинки по каждому классу
fig, axs = plt.subplots(1, 10, figsize=(12, 3)) #Создаем полотно из 10 графиков
for i in range(10): #Проходим по классам от 0 до 9
    label_indexes = np.where(y_train==i)[0] #Получаем список из индексов положений класса i в y_train
    index = random.choice(label_indexes) #Случайным образом выбираем из списка индекс
    img = x_train[index] #Выбираем из x_train нужное изображение
    axs[i].imshow(Image.fromarray(img)) #Отображаем изображение i-ым графиков
# plt.show() #Показываем изображения

x_train = x_train / 255
x_test = x_test / 255

y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

inputs = keras.Input(shape=(32, 32, 3), name="img")
x = layers.Conv2D(32, 3, activation="relu")(inputs)
x = layers.Conv2D(64, 3, activation="relu")(x)
block_1_output = layers.MaxPooling2D(3)(x)

x = layers.Conv2D(64, 3, activation="relu",
padding="same")(block_1_output)
x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
block_2_output = layers.add([x, block_1_output])

x = layers.Conv2D(64, 3, activation="relu",
padding="same")(block_2_output)
x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
block_3_output = layers.add([x, block_2_output])

x = layers.Conv2D(64, 3, activation="relu")(block_3_output)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(256, activation="relu")(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(10, activation='softmax')(x)

model = keras.Model(inputs, outputs, name="my_resnet")

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history=model.fit(x_train, y_train, batch_size=64, 
                  epochs=18, validation_split=0.2,
                  callbacks=[tensorboard_callback])

model.summary()

# keras.utils.plot_model(model, show_shapes=True)

plt.plot(history.history['accuracy'],
         label='Доля верных ответов на обучающем наборе')
plt.plot(history.history['val_accuracy'],
         label='Доля верных ответов на проверочном наборе')
plt.xlabel('Эпоха обучения')
plt.ylabel('Доля верных ответов')
plt.legend()
plt.show()

# График потерь отдельно
plt.plot(history.history['loss'],
         label='Потери на обучающем наборе')
plt.plot(history.history['val_loss'],
         label='Потери на проверочном наборе')
plt.xlabel('Эпоха обучения')
plt.ylabel('Потери')
plt.title('График потерь модели')
plt.legend()
plt.show()

print(model.evaluate(x_test, y_test))

prediction = model.predict(x_test)
print(prediction.shape)

#пример: Выбираем номер изображения
n = 27
#Выводим на экран картинку
plt.imshow(Image.fromarray((x_test[n]*255).astype(np.uint8)).convert('RGBA'))
#Выводим на экран результаты
print("Выход сети: ", prediction[n])
print("Распознанный образ: ", np.argmax(prediction[n]))
print("Верный ответ: ", y_test[n])
print("Распознанный образ на картинке: ", classes[np.argmax(prediction[n])])

xTestReal5 = []
yTestReal5 = []

for i in range(5):
    img_path = f"./4_2/images/{str(i)}.jpg"
    xTestReal5.append(np.asarray(image.load_img(img_path, color_mode='rgb', target_size=(32,32))))
    yTestReal5.append(i)

xTestReal5 = np.array(xTestReal5)
yTestReal5 = np.array(yTestReal5)

fig, axs = plt.subplots(1, 5, figsize=(12,3))
for i in range(5):
    axs[i].imshow((xTestReal5[i] * 255).astype(np.uint8))
    axs[i].set_title(f"Image {i+1}")
    axs[i].axis('off')
plt.show()

prediction = model.predict(xTestReal5)
for i in range(5):
    print(f"Изображение {i}:")
    print(f"Распознанный образ: {classes[np.argmax(prediction[i])]}")
    print(f"Верный ответ: {classes[yTestReal5[i]]}")  # Добавлен индекс [i]
    print(f"Уверенность: {np.max(prediction[i]):.2f}")
    print()