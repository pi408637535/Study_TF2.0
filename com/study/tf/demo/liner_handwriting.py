import tensorflow as tf
from tensorflow.keras import datasets
from tensorflow import keras
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'

def preprocess(x, y):
    x = tf.cast(x, dtype=tf.float32) /255
    y = tf.cast(y, dtype=tf.int32)
    return x,y


if __name__ == '__main__':

    (x,y),(x_test, y_test) = datasets.mnist.load_data()

    x = tf.convert_to_tensor(x, dtype=tf.float32)
    y = tf.convert_to_tensor(y, dtype=tf.int32)

    train_db = tf.data.Dataset.from_tensor_slices((x,y) ).batch(32)
    train_db = train_db.map(preprocess)

    test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)
    test_db = test_db.map(preprocess)

    db_iter = iter(train_db)
    sample = next(db_iter)
    print("batch:", sample[0].shape, sample[1].shape)

    model = keras.Sequential([
        keras.layers.Dense(256, activation=tf.nn.relu),
        keras.layers.Dense(128, activation=tf.nn.relu),
        keras.layers.Dense(64, activation=tf.nn.relu),
        keras.layers.Dense(32, activation=tf.nn.relu),
        keras.layers.Dense(10)
    ])
    model.build(input_shape=[None, 28 * 28])
    model.summary()
    optimizer = keras.optimizers.Adam(lr = 1e-3)
    for epoch in range(3):
        for step,(x,y) in enumerate(train_db):
            x = tf.reshape(x, [-1, 28 * 28]) #batch,28 * 28

            with tf.GradientTape() as tape:
                logits = model(x)
                y_onehot = tf.one_hot(y, depth=10)

                #tf中的loss是针对每个instance来计算的
                loss_ce = tf.reduce_mean( tf.losses.categorical_crossentropy(y_onehot, logits, from_logits=True))
                loss_mse = tf.reduce_mean( tf.losses.mse(y_onehot, logits) )


            grads = tape.gradient(loss_ce, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))


            if step % 100 == 0:
                print(epoch, step, "loss:", float(loss_ce), float(loss_mse))

        for x,y in test_db:





