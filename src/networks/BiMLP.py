import tensorflow as tf


class BiMLP:
    def __init__(self, emb_dim):
        self.emb_dim = emb_dim
        self.bilinear_weights = tf.Variable(tf.random_normal(
            shape=[emb_dim, 2*emb_dim, 2*emb_dim]
        ), name='biW')
        self.mlp_weight = tf.Variable(tf.random_normal(
            shape=[emb_dim, 1]
        ), name='mlpW')
        self.mlp_bias = tf.Variable(tf.random_normal(shape=[1]), name='mlpb')

    def forward(self, jd_data, cv_data):
        x = tf.map_fn(
            lambda x: self.bilinear(jd_data, cv_data, x),
            self.bilinear_weights
        )
        x = tf.transpose(x, perm=[1, 0])
        x = tf.sigmoid(x)
        x = tf.matmul(x, self.mlp_weight)
        x = tf.add(x, self.mlp_bias)
        x = tf.sigmoid(x, name='predict')
        return x

    def bilinear(self, jd_data, cv_data, weights):
        x = tf.matmul(jd_data, weights)
        x = tf.multiply(x, cv_data)
        x = tf.reduce_sum(x, axis=1)
        return x


if __name__ == '__main__':
    import numpy as np
    jd = np.random.randint(10, size=[128, 200])
    cv = np.random.randint(10, size=[128, 200])

    JD = tf.placeholder(dtype=tf.float32, shape=[None, 200])
    CV = tf.placeholder(dtype=tf.float32, shape=[None, 200])
    bimlp = BiMLP(100)
    out = bimlp.forward(JD, CV)

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        out = sess.run(out, feed_dict={JD: jd, CV: cv})
    print(out.shape)
