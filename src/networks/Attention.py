import tensorflow as tf
import numpy as np


class Attention:
    def __init__(self, doc_len):
        self.doc_len = doc_len

    def forward(self, jd_data, cv_data, inf_masks):
        """
        :param cv_data: 3d array, batch_size * doc_len * emb_dim
        :param jd_data: 3d array, batch_size * doc_len * emb_dim
        :return:
        """
        attention_weights = tf.matmul(jd_data, tf.transpose(cv_data, [0, 2, 1]))
        attention_weights = attention_weights + inf_masks

        cv_weights = tf.nn.softmax(attention_weights, axis=2)
        jd_context = tf.matmul(cv_weights, cv_data)
        jd_cat = tf.concat([jd_data, jd_context], axis=2)
        jd_cat = tf.reduce_max(jd_cat, reduction_indices=1)

        jd_weights = tf.nn.softmax(attention_weights, axis=1)
        cv_context = tf.matmul(jd_weights, jd_data)
        cv_cat = tf.concat([cv_data, cv_context], axis=2)
        cv_cat = tf.reduce_max(cv_cat, reduction_indices=1)

        return jd_cat, cv_cat

    def get_inf_mask(self, jd_len, cv_len):
        inf_mask = np.zeros(shape=[self.doc_len, self.doc_len])
        inf_mask[:jd_len] = float('-inf')
        inf_mask[:, cv_len:] = float('-inf')
        return inf_mask

    def get_zero_mask(self, jd_len, cv_len):
        zero_mask = np.ones(shape=[self.doc_len, self.doc_len])
        zero_mask[:jd_len] = 0
        zero_mask[:, cv_len:] = 0
        return zero_mask

    def get_masks(self, jd_lens, cv_lens):
        inf_masks = [self.get_inf_mask(cv_len, jd_len)
                     for cv_len, jd_len in zip(cv_lens, jd_lens)]
        inf_masks = np.array(inf_masks)
        zero_masks = [self.get_zero_mask(cv_len, jd_len)
                      for cv_len, jd_len in zip(cv_lens, jd_lens)]
        zero_masks = np.array(zero_masks)
        return inf_masks, zero_masks


if __name__ == '__main__':
    from CNN import textCNN
    data1 = np.random.randint(200, size=[10, 25, 50])
    data2 = np.random.randint(200, size=[10, 25, 50])

    X1 = tf.placeholder(dtype=tf.int32, shape=[None, 25, 50])
    X2 = tf.placeholder(dtype=tf.int32, shape=[None, 25, 50])
    INF_MASK = tf.placeholder(dtype=tf.float32, shape=[None, 25, 25])
    ZERO_MASK = tf.placeholder(dtype=tf.float32, shape=[None, 25, 25])

    cnn1 = textCNN(200, 100, 25, 50)
    Y1 = cnn1.forward(X1)

    cnn2 = textCNN(200, 100, 25, 50)
    Y2 = cnn2.forward(X2)

    l1 = np.random.randint(25, size=[10])
    l2 = np.random.randint(25, size=[10])
    attention = Attention(25)
    inf_mask, zero_mask = attention.get_masks(l1, l2)
    Z = attention.forward(Y1, Y2, INF_MASK)

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        out1, out2 = sess.run(Z, feed_dict={
            X1: data1,
            X2: data2,
            INF_MASK: inf_mask,
            ZERO_MASK: zero_mask
        })
    print(out1.shape)
    print(out2.shape)



