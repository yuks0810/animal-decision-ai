# トレーニング実行コード
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
# from keras.utils import np_utils
from tensorflow.python.keras.utils import np_utils
# https://github.com/keras-team/keras/issues/8838#issuecomment-557522458
import numpy as np

classes = ["monkey", "boar", "crow"]
num_classes = len(classes)
image_size = 50

# メインの関数を定義する
# データの正規化：最大値で割ることで、RGBの各値が0~1の中に収まるようになる
def main():
    X_train, X_test, y_train, y_test = np.load("./animal.npy",allow_pickle=True)
    X_train = X_train.astype("float") / 256 # 256階調のRGBデータがある
    X_test = X_test.astype("float") / 256 # 256階調のRGBデータがある

    # one-hot-vector化
    y_train = np_utils.to_categorical(y_train, num_classes)
    y_test = np_utils.to_categorical(y_test, num_classes)

    model = model_train(X_train, y_train)
    model_eval(model, X_test, y_test)

"""
one-hot-vector: 正解値は1,他は0
[0,1,2]を[1,0,0], [0,1,0], [0,0,1]に変換
0 = monkey
1 = crow
2 = boar
"""


def model_train(X, y):
    model = Sequential()
    model.add(Conv2D(32, (3,3), padding='same', input_shape=X.shape[1:]))
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

    opt = tf.keras.optimizers.RMSprop(lr=0.0001, decay=1e-6) # import tensorflowしないと使えなかった

    model.compile(loss='categorical_crossentropy',
                optimizer=opt,
                metrics=['accuracy'])


    model.fit(X, y, batch_size=32, epochs=40)

    # モデルの保存
    model.save('./animal_cnn.h5')

    return model

def model_eval(model, x, y):
    scores = model.evaluate(x, y, verbose=1)
    print('Test Loss: ', scores[0])
    print('Test Accuracy:', scores[1])

if __name__=="__main__":
    main()