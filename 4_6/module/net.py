from keras.models import Sequential
from keras.layers import BatchNormalization,Conv2D, MaxPooling2D, \
Activation,Flatten, Dropout, Dense
from keras import backend as K

class NetModel:
    @staticmethod
    def build(width, height, depth, classes):
        # для порядка каналов “channel_last” и размер канала
        #Cначала инициализируем последовательную (Sequential) модель.
        model = Sequential()
        #Затем определяем порядок каналов и размер входного изображения.
        inputShape = (height, width, depth)
        chanDim = -1
        # если используем порядок "channels first", обновляем
        # входное изображение и размер канала
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
            chanDim = 1

        #В этом блоке добавляются слои CONV => RELU => POOL.
        #Первый слой CONV имеет 32 фильтра размером 3х3.
        model.add(Conv2D(32, (3, 3), padding="same",
        input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(64, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(64, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        #(CONV => RELU) * 3 => POOL:
        model.add(Conv2D(128, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(128, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(128, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        # классификатор softmax
        model.add(Dense(classes))
        model.add(Activation("softmax"))

        # возвращаем собранную архитектуру нейронной сети
        return model
    
