import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import sys
sys.path.append('../src/')
import tensorflow as tf
import numpy as np
from networks.CNN import TextCNN


if __name__ == '__main__':
    cnn = TextCNN(200, 100, 25, 50)
    data = np.random.randint(200, size=[10, 25, 50])
    X = tf.placeholder(dtype=tf.int32, shape=[None, 25, 50])

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        out = sess.run(cnn.forward(X), feed_dict={X: data})
    print(out.shape)