import tensorflow as tf
from .boardUtils import var_summaries
from .CNN import TextCNN
from .Attention import Attention
from .BiMLP import BiMLP


class MatchModel:
    def __init__(self, n_word, emb_dim, sent_len, doc_len, emb_pretrain=[]):
        self.emb_dim = emb_dim
        self.sent_len = sent_len
        self.doc_len = doc_len
        self.n_word = n_word
        self.emb_pretrain = emb_pretrain

        # with tf.name_scope('jd_cnn'):
        #     self.jd_cnn = TextCNN(n_word, emb_dim, doc_len, sent_len, emb_pretrain)
        # with tf.name_scope('cv_cnn'):
        #     self.cv_cnn = TextCNN(n_word, emb_dim, doc_len, sent_len, emb_pretrain)
        # with tf.name_scope('attention'):
        #     self.att = Attention(doc_len)
        # with tf.name_scope('mlp'):
        #     self.mlp = BiMLP(emb_dim)

        self.jd = tf.placeholder(dtype=tf.int32, shape=[None, doc_len, sent_len], name='jd')
        self.cv = tf.placeholder(dtype=tf.int32, shape=[None, doc_len, sent_len], name='cv')
        self.inf_mask = tf.placeholder(dtype=tf.float32, shape=[None, doc_len, doc_len], name='inf_mask')
        self.zero_mask = tf.placeholder(dtype=tf.float32, shape=[None, doc_len, doc_len], name='zero_mask')
        self.predict = self.forward()

    def forward(self):
        with tf.name_scope('jd_cnn'):
            self.jd_cnn = TextCNN(self.n_word, self.emb_dim, self.doc_len, self.sent_len, self.emb_pretrain)
            jd = self.jd_cnn.forward(self.jd)
        with tf.name_scope('cv_cnn'):
            self.cv_cnn = TextCNN(self.n_word, self.emb_dim, self.doc_len, self.sent_len, self.emb_pretrain)
            cv = self.cv_cnn.forward(self.cv)
        # =========================================================================
        with tf.name_scope('attention'):
            self.att = Attention(self.doc_len)
        #     jd, cv = self.att.forward(jd, cv, self.inf_mask, self.zero_mask)

        # =========================================================================
        with tf.name_scope('mlp'):
            self.mlp = BiMLP(self.emb_dim)
            score = self.mlp.forward(jd, cv)
            var_summaries(score)
            return score

    def get_masks(self, jd_data_np, cv_data_np):
        return self.att.get_masks(jd_data_np, cv_data_np)

    def loss_function(self):
        label = tf.placeholder(dtype=tf.int32, shape=None, name='label')
        loss = tf.losses.log_loss(label, self.predict)
        tf.summary.scalar('log loss', loss)
        # loss = tf.losses.mean_squared_error(label, self.predict)
        # auc, _ = tf.metrics.auc(label, self.predict)
        # return loss, auc
        return loss

