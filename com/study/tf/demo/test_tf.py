import tensorflow as tf
from tensorflow import keras

if __name__ == '__main__':
    x = tf.random.normal([2,4])
    model = keras.Sequential([
        keras.layers.Dense(2, activation="relu"),
        keras.layers.Dense(2, activation="relu"),
        keras.layers.Dense(2),
    ])

    model.build(input_shape=[None, 4])
    model.summary()

