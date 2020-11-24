import tensorflow as tf
from tensorflow.keras import datasets
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'

(x,y),_ = datasets.mnist.load_data()

x = tf.convert_to_tensor(x, dtype=tf.float32)
y = tf.convert_to_tensor(y, dtype=tf.int32)

train_db = tf.data.Dataset.from_tensor_slices((x,y) ).batch(32)
w1 = tf.Variable( tf.random.truncated_normal([28 * 28 , 256], stddev=0.1) )
w2 = tf.Variable( tf.random.truncated_normal([256, 64],stddev=0.1))
w3 = tf.Variable( tf.random.truncated_normal([64, 10], stddev=0.1 ) )
lr = 10 ** (-3)

for step, (x,y) in enumerate(train_db):
    x = tf.reshape(x, [-1, 28 * 28])


    #caculate gradient
    with tf.GradientTape() as tape:

        h1 = x@w1
        h2 = h1@w2
        y_pre = h2@w3

        #y_pre:batch,10
        y_onehot = tf.one_hot(y, depth=10)

        #y:batch
        #y_onehot:batch,10
        loss = tf.square(y_onehot - y_pre)
        loss = tf.reduce_mean(loss)

    grads = tape.gradient(loss, [w1, w2, w3])
    # w1 = w1 - lr * grads[0] # don't assign new variable
    # w2 w2 - lr * grads[1]
    # w3 = w3 - lr * grads[2]
    w1.assign_sub(lr * grads[0])
    w2.assign_sub(lr * grads[1])
    w3.assign_sub(lr * grads[2])

    if step % 100:
        print("step={0}, loss = {1}".format(str(step), str(loss)))

