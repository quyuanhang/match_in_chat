import tensorflow as tf
from .boardUtils import vars_summaries


class CNN:
    def __init__(self, emb_dim, sent_len):
        self.sent_len = sent_len
        self.weights = {
            'wc1': tf.Variable(tf.random_normal([5, emb_dim, 1, emb_dim]), name='cnn_w1', dtype=tf.float32),
            'wc2': tf.Variable(tf.random_normal([3, emb_dim, 1, emb_dim]), name='cnn_w2', dtype=tf.float32)
        }
        self.bias = {
            'bc1': tf.Variable(tf.random_normal([emb_dim]), name='cnn_b1', dtype=tf.float32),
            'bc2': tf.Variable(tf.random_normal([emb_dim]), name='cnn_b2', dtype=tf.float32)
        }
        vars_summaries(self.weights['wc1'], self.weights['wc2'], self.bias['bc1'], self.bias['bc2'])

    @staticmethod
    def conv2d(x, W, b, strides, padding=[0, 0]):
        p0 = padding[0]
        p1 = padding[1]
        x = tf.pad(x, [[0, 0], [p0, p0], [p1, p1], [0, 0]])
        x = tf.nn.conv2d(x, W, strides=[1, *strides, 1], padding='VALID')
        x = tf.nn.bias_add(x, b)
        return tf.nn.relu(x)

    def forward(self, x):
        '''
        :param x: 4d array: (batch * doc_len) * sent_len * emb_dim * 1
        :return: 4d array: (batch * doc_len) * 1 * 1 * emb_dim
        '''
        x = self.conv2d(
            x=x,
            W=self.weights['wc1'],
            b=self.bias['bc1'],
            strides=[1, 1],
            padding=[2, 0]
        )
        x = tf.transpose(x, [0, 1, 3, 2])
        x = self.conv2d(
            x=x,
            W=self.weights['wc2'],
            b=self.bias['bc2'],
            strides=[1, 1],
            padding=[1, 0]
        )
        x = tf.nn.max_pool(
            value=x,
            ksize=[1, self.sent_len, 1, 1],
            strides=[1, 1, 1, 1],
            padding='VALID'
        )
        return x


class TextCNN:
    def __init__(self, n_word, emb_dim, doc_len, sent_len, emb_weights=[]):
        self.doc_len = doc_len
        self.sent_len = sent_len
        self.emb_dim = emb_dim
        self.cnn = CNN(emb_dim, sent_len)
        if len(emb_weights) == 0:
            self.emb_weights = tf.Variable(tf.random_normal([n_word, emb_dim]), name='emb_w', dtype=tf.float32)
        else:
            self.emb_weights = tf.Variable(emb_weights, name='emb_w', dtype=tf.float32)
            self.emb_weights = tf.stop_gradient(self.emb_weights, name='emb_w_freeze')

    # def forward(self, x):
    #     '''
    #     :param x: 3d array, batch_size * doc_len * set_len
    #     :return: 3d array, batch_size * doc_len * emb_dim
    #     '''
    #     x = tf.nn.embedding_lookup(self.emb_weights, x)
    #     x = tf.reshape(x, shape=[-1, self.sent_len, self.emb_dim, 1])
    #     x = self.cnn.forward(x)
    #     x = tf.reshape(x, shape=[-1, self.doc_len, self.emb_dim])
    #     return x

    def forward(self, x):
        x = tf.nn.embedding_lookup(self.emb_weights, x)
        x = tf.reshape(x, shape=[-1, self.doc_len * self.sent_len, self.emb_dim])
        x = tf.reduce_sum(x, axis=1)
        # x = tf.reshape(x, shape=[-1, self.sent_len, self.emb_dim, 1])
        # x = self.cnn.forward(x)
        # x = tf.reshape(x, shape=[-1, self.doc_len, 1, self.emb_dim])
        # x = tf.nn.max_pool(
        #     value=x,
        #     ksize=[1, self.doc_len, 1, 1],
        #     strides=[1, 1, 1, 1],
        #     padding='VALID'
        # )
        # x = tf.squeeze(x)
        with tf.variable_scope('bn'):
            axis = list(range(len(x.get_shape()) - 1))
            mean, variance = tf.nn.moments(x, axis)
            x = tf.nn.batch_normalization(x, mean, variance, offset=0, scale=1, variance_epsilon=1e-5)
        return x



