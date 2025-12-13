import cv2
from matplotlib import pyplot as plt

def plot_img(img):
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(rgb)
    plt.axis('off')  # Отключаем оси для лучшего отображения
    plt.show()  # Это ключевая строка!

car = cv2.imread("./4_12/images/car.jpg")
plot_img(car)

classifier = cv2.CascadeClassifier()
classifier.load("./4_12/haarcascade_russian_plate_number.xml")

plates = classifier.detectMultiScale(car)
with_indicators = car
for plate in plates:
    x, y, width, height = plate
    with_indicators = cv2.rectangle(with_indicators, (x,y),
                                    (x+width, y+height),
                                    (0,0,255), 5)

plot_img(with_indicators)

car = cv2.imread("./4_12/images/lada_vesta.jpg")
plot_img(car)
plates = classifier.detectMultiScale(car)
with_indicators = car
for plate in plates:
    x, y, width, height = plate
    with_indicators = cv2.rectangle(with_indicators, (x,y),
                                    (x+width, y+height),
                                    (0,0,255), 5)
plot_img(with_indicators)