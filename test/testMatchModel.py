import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import sys
sys.path.append('../src/')
import tensorflow as tf
import numpy as np
from networks import MatchModel

if __name__ == '__main__':

    model = MatchModel(200, 100, 50, 25)

    data1 = np.random.randint(200, size=[10, 25, 50])
    data2 = np.random.randint(200, size=[10, 25, 50])
    inf_mask, zero_mask = model.get_masks(data1, data2)
    label = np.random.randint(2, size=[10])

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        writer = tf.summary.FileWriter("board", sess.graph)
        out = sess.run(model.predict, feed_dict={
            model.jd: data1,
            model.cv: data2,
            model.inf_mask: inf_mask,
            model.zero_mask: zero_mask,
            # 'label:0': label
        })
    writer.close()
    print(out)


