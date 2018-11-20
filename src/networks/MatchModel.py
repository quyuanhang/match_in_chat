import tensorflow as tf

from .CNN import textCNN
from .Attention import Attention
from .BiMLP import BiMLP


class MatchModel:
    def __init__(self, n_word, emb_dim, sent_len, doc_len, emb_pretrain=[]):
        self.emb_dim = emb_dim
        self.sent_len = sent_len
        self.doc_len = doc_len

        self.jd_cnn = textCNN(n_word, emb_dim, doc_len, sent_len, emb_pretrain)
        self.cv_cnn = textCNN(n_word, emb_dim, doc_len, sent_len, emb_pretrain)
        self.att = Attention(doc_len)
        self.mlp = BiMLP(emb_dim)

        self.jd = tf.placeholder(dtype=tf.int32, shape=[None, doc_len, sent_len], name='jd')
        self.cv = tf.placeholder(dtype=tf.int32, shape=[None, doc_len, sent_len], name='cv')
        self.inf_mask = tf.placeholder(dtype=tf.float32, shape=[None, doc_len, doc_len], name='inf_mask')
        self.zero_mask = tf.placeholder(dtype=tf.float32, shape=[None, doc_len, doc_len], name='zero_mask')
        self.predict = self.forward()
        # self.loss, self.auc, self.auc_op = self.loss_function()

    def forward(self):
        with tf.name_scope('jd_cnn'):
            jd = self.jd_cnn.forward(self.jd)
        with tf.name_scope('cv_cnn'):
            cv = self.cv_cnn.forward(self.cv)
        with tf.name_scope('attention'):
            jd, cv = self.att.forward(jd, cv, self.inf_mask, self.zero_mask)
        with tf.name_scope('mlp'):
            score = self.mlp.forward(jd, cv)
        return score

    def get_masks(self, jd_data_np, cv_data_np):
        return self.att.get_masks(jd_data_np, cv_data_np)

    def loss_function(self):
        label = tf.placeholder(dtype=tf.int32, shape=None, name='label')
        loss = tf.losses.log_loss(label, self.predict)
        auc, _ = tf.metrics.auc(label, self.predict)
        return loss, auc


