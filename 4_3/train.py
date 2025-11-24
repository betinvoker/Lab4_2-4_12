import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from io import BytesIO
from PIL import Image

with open('./4_3/images/car.png', 'rb') as f:
    img_data = f.read()

img = Image.open(BytesIO(img_data))
plt.imshow(img)
plt.show()

img = np.array(img)
x = keras.applications.vgg16.preprocess_input(img)
print(x.shape)
x = np.expand_dims(x, axis=0)

model = keras.applications.VGG16()

res = model.predict(x)
print(np.argmax(res))

with open('./4_3/datasets/imagenet1000_clsidx_to_labels.txt') as f:
    data = f.readlines()

print(data[np.argmax(res)])

with open('./4_3/images/apple.png', 'rb') as f:
    img_data = f.read()

img = Image.open(BytesIO(img_data))
img = img.resize((224, 224))
plt.imshow(img)
plt.show()

img = np.array(img)
x = keras.applications.vgg16.preprocess_input(img)
print(x.shape)
x = np.expand_dims(x, axis=0)

model = keras.applications.VGG16()

res = model.predict(x)
print(np.argmax(res))

print(data[np.argmax(res)])

with open('./4_3/images/rabbit.png', 'rb') as f:
    img_data = f.read()

img = Image.open(BytesIO(img_data))
img = img.resize((224, 224))
plt.imshow(img)
plt.show()

img = np.array(img)
x = keras.applications.vgg16.preprocess_input(img)
print(x.shape)
x = np.expand_dims(x, axis=0)

model = keras.applications.VGG16()

res = model.predict(x)
print(np.argmax(res))

print(data[np.argmax(res)])