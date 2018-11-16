import tensorflow as tf
from .CNN import textCNN
from .Attention import Attention
from .BiMLP import BiMLP


class MatchModel:
    def __init__(self, n_word, emb_dim, sent_len, doc_len):
        self.emb_dim = emb_dim
        self.sent_len = sent_len
        self.doc_len = doc_len

        self.jd_cnn = textCNN(n_word, emb_dim, doc_len, sent_len)
        self.cv_cnn = textCNN(n_word, emb_dim, doc_len, sent_len)
        self.att = Attention(doc_len)
        self.mlp = BiMLP(emb_dim)

    def forward(self, jd, cv, inf_mask, zero_mask):
        jd = self.jd_cnn.forward(jd)
        cv = self.cv_cnn.forward(cv)
        jd, cv = self.att.forward(jd, cv, inf_mask, zero_mask)
        score = self.mlp.forward(jd, cv)
        return score

    def get_masks(self, jd_data_np, cv_data_np):
        return self.att.get_masks(jd_data_np, cv_data_np)


if __name__ == '__main__':
    import numpy as np

    X1 = tf.placeholder(dtype=tf.int32, shape=[None, 25, 50])
    X2 = tf.placeholder(dtype=tf.int32, shape=[None, 25, 50])
    INF_MASK = tf.placeholder(dtype=tf.float32, shape=[None, 25, 25])
    ZERO_MASK = tf.placeholder(dtype=tf.float32, shape=[None, 25, 25])

    model = MatchModel(200, 100, 50, 25)
    Z = model.forward(X1, X2, INF_MASK, ZERO_MASK)

    data1 = np.random.randint(200, size=[10, 25, 50])
    data2 = np.random.randint(200, size=[10, 25, 50])
    inf_mask, zero_mask = model.get_masks(data1, data2)

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        out = sess.run(Z, feed_dict={
            X1: data1,
            X2: data2,
            INF_MASK: inf_mask,
            ZERO_MASK: zero_mask
        })
    print(out)
