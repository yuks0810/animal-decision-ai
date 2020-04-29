import tensorflow as tf
import sys
from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.python.keras.utils import np_utils
import numpy as np
from PIL import Image
from sklearn import model_selection

classes = ["monkey", "boar", "crow"]
num_classes = len(classes)
image_size = 50

def build_model():
    model = Sequential()
    model.add(Conv2D(32, (3,3), padding='same', input_shape=(50,50,3)))
    model.add(Activation('relu'))
    model.add(Conv2D(32,(3,3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64,(3,3),padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64,(3,3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    # opt = tf.keras.optimizers.RMSprop(lr=0.0001, decay=1e-6) # import tensorflowしないと使えなかった
    opt = tf.keras.optimizers.Adam()

    model.compile(loss='categorical_crossentropy',
                optimizer=opt,
                metrics=['accuracy'])

    # モデルのロード
    # model = tf.keras.models('./animal_cnn_aug.h5')
    model = load_model('./animal_cnn_aug.h5')

    return model

def main():
    # python predict.py filename <- これが引数
    # filenameは２番目の引数
    # ゼロから数えると１番目
    image = Image.open(sys.argv[1])
    imaage = image.convert('RGB')
    image = image.resize((image_size, image_size))
    data = np.asarray(image)
    X = []
    X.append(data)
    X = np.array(X)
    model = build_model()

    result = model.predict([X])[0]
    predicted = result.argmax() # 一番値の大きい配列の添字を返す
    percentage = int(result[predicted] * 100)
    print("{0} ({1} %)".format(classes[predicted], percentage))
    print(result)

if __name__ == "__main__":
    main()
else:
    print("NOT")