import tensorflow as tf
from tensorflow import keras
import numpy as np
from .boardUtils import vars_summaries


class MatchModel:
    def __init__(self, n_word:int, emb_dim:int, doc_len:int, sent_len:int, attention=False, bil=False, reduce='mean', emb_pretrain=[]):
        self.emb_dim = emb_dim
        self.sent_len = sent_len
        self.doc_len = doc_len
        self.n_word = n_word
        self.att = attention
        self.reduce = reduce

        if len(emb_pretrain) > 0:
            def myinit(*args, **kwargs):
                return tf.convert_to_tensor(emb_pretrain, dtype=tf.float32)
            self.emb_init = myinit
        else:
            self.emb_init = 'RandomNormal'

        self.jd = tf.placeholder(dtype=tf.int32, shape=[None, doc_len, sent_len], name='jds')
        self.cv = tf.placeholder(dtype=tf.int32, shape=[None, doc_len, sent_len], name='cvs')
        self.inf_masks = tf.placeholder(dtype=tf.float32, shape=[None, doc_len, doc_len], name='inf_masks')
        self.zero_masks = tf.placeholder(dtype=tf.float32, shape=[None, doc_len, doc_len], name='zero_masks')
        self.label = tf.placeholder(dtype=tf.int32, shape=[None], name='labels')

        with tf.variable_scope('jd_cnn'):
            jd = self.text_cnn(self.jd)
        with tf.variable_scope('cv_cnn'):
            cv = self.text_cnn(self.cv)
        if attention:
            with tf.variable_scope('attention'):
                jd, cv = self.attention(jd, cv)
        with tf.variable_scope('classifier'):
            if bil:
                self.predict = self.bilinear_classifier(jd, cv)
            else:
                self.predict = self.classifier(jd, cv)
        with tf.variable_scope('loss'):
            self.loss = self.loss_function()
            self.auc = self.metric()

    def text_cnn(self, x:tf.Tensor):
        layers = keras.Sequential([
            keras.layers.Reshape(
                target_shape=[self.doc_len * self.sent_len]),
            keras.layers.Embedding(
                input_dim=self.n_word,
                output_dim=self.emb_dim,
                embeddings_initializer=self.emb_init),
            keras.layers.Reshape(
                target_shape=[self.doc_len * self.sent_len, self.emb_dim, 1]),
            keras.layers.ZeroPadding2D(
                padding=(2, 0)),
            keras.layers.Conv2D(
                filters=self.emb_dim,
                # kernel_regularizer=keras.regularizers.l2(0.0001),
                kernel_size=[5, self.emb_dim]),
            keras.layers.Permute(
                dims=[1, 3, 2]),
            keras.layers.ZeroPadding2D(
                padding=(1, 0)),
            keras.layers.Conv2D(
                filters=self.emb_dim,
                # kernel_regularizer=keras.regularizers.l1_l2(),
                # kernel_regularizer=keras.regularizers.l2(0.0001),
                kernel_size=[3, self.emb_dim]),
            keras.layers.Reshape(
                target_shape=[self.doc_len, self.sent_len, self.emb_dim]),
        ])
        x = layers(x)
        if self.att:
            if self.reduce == 'mean':
                x = tf.reduce_mean(x, axis=2)
            else:
                x = tf.reduce_max(x, axis=2)
        else:
            x = tf.reduce_mean(x, axis=[1, 2])
            x = tf.nn.relu(x)
        return x

    def get_inf_mask(self, jd_len, cv_len):
        inf_mask = np.zeros(shape=[self.doc_len, self.doc_len])
        inf_mask[jd_len:] = -1e10
        inf_mask[:, cv_len:] = -1e10
        return inf_mask

    def get_zero_mask(self, jd_len, cv_len):
        zero_mask = np.ones(shape=[self.doc_len, self.doc_len])
        zero_mask[jd_len:] = 0
        zero_mask[:, cv_len:] = 0
        return zero_mask

    def get_masks(self, jd_data_np, cv_data_np):
        """
        :param jd_data_np: 3d ndarray: batch * doc_len * emb_dim,
        :param cv_data_np: same
        :return: 3d ndarray: batch * doc_len * doc_len
        """
        jd_lens = [sum(jd.any(axis=1)) for jd in jd_data_np]
        cv_lens = [sum(cv.any(axis=1)) for cv in cv_data_np]
        inf_masks = [self.get_inf_mask(cv_len, jd_len)
                     for cv_len, jd_len in zip(cv_lens, jd_lens)]
        inf_masks = np.array(inf_masks)
        zero_masks = [self.get_zero_mask(cv_len, jd_len)
                      for cv_len, jd_len in zip(cv_lens, jd_lens)]
        zero_masks = np.array(zero_masks)
        return inf_masks, zero_masks

    def attention(self, jd_data, cv_data):
        """
        :param cv_data: 3d array, batch_size * doc_len * emb_dim
        :param jd_data: 3d array, batch_size * doc_len * emb_dim
        :return:
        """
        attention_weights = tf.matmul(jd_data, tf.transpose(cv_data, [0, 2, 1]))
        attention_weights = attention_weights + self.inf_masks

        cv_weights = tf.multiply(tf.nn.softmax(attention_weights, axis=2), self.zero_masks)
        jd_context = tf.matmul(cv_weights, cv_data)
        jd_cat = tf.concat([jd_data, jd_context], axis=2)
        if self.reduce == 'mean':
            jd_cat = tf.reduce_mean(jd_cat, reduction_indices=1)
        else:
            jd_cat = tf.reduce_max(jd_cat, reduction_indices=1)
        jd_cat = tf.nn.relu(jd_cat)

        jd_weights = tf.multiply(tf.nn.softmax(attention_weights, axis=1), self.zero_masks)
        jd_weights = tf.transpose(jd_weights, perm=[0,2,1])
        cv_context = tf.matmul(jd_weights, jd_data)
        cv_cat = tf.concat([cv_data, cv_context], axis=2)
        if self.reduce == 'mean':
            cv_cat = tf.reduce_mean(cv_cat, reduction_indices=1)
        else:
            cv_cat = tf.reduce_max(cv_cat, reduction_indices=1)
        cv_cat = tf.nn.relu(cv_cat)

        return jd_cat, cv_cat

    def classifier(self, jd, cv):
        x = keras.layers.Concatenate()([jd, cv])
        # x = keras.layers.Dense(
        #     self.emb_dim // 2,
        #     activation='sigmoid',
        #     kernel_regularizer=keras.regularizers.l1_l2(),
        # )(x)
        x = keras.layers.Dense(
            1,
            activation='sigmoid',
            name='prediction',
            # kernel_regularizer=keras.regularizers.l1_l2(),
            # kernel_regularizer=keras.regularizers.l2(0.0001),
        )(x)
        return x

    def bilinear(self, jd_data, cv_data, weights):
        x = tf.matmul(jd_data, weights)
        x = tf.multiply(x, cv_data)
        x = tf.reduce_sum(x, axis=1)
        return x

    def bilinear_classifier(self, jd, cv):
        emb_dim = int(jd.shape[-1])
        bilinear_weights = tf.get_variable(
            name='biW',
            initializer=tf.random_normal(shape=[emb_dim, emb_dim, emb_dim]),
        )
        x = tf.map_fn(
            lambda w: self.bilinear(jd, cv, w),
            bilinear_weights
        )
        x = tf.transpose(x, perm=[1,0])
        layers = keras.Sequential([
            keras.layers.Activation('relu'),
            keras.layers.Dense(1, activation='sigmoid')
        ])
        x = layers(x)
        return x

    def loss_function(self):
        predict = tf.squeeze(self.predict)
        loss = tf.losses.log_loss(self.label, predict)
        return loss

    def metric(self):
        predict = tf.squeeze(self.predict)
        metric = tf.metrics.auc(self.label, predict)
        return metric


if __name__ == '__main__':
    import argparse
    from tqdm import tqdm
    from utils import dataSet
    from sklearn import metrics

    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', default='-1')
    parser.add_argument('--datain', nargs='?', default='interview')
    parser.add_argument('--dataout', default='interview')
    parser.add_argument('--emb_dim', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=128)
    # word2vec arguments
    parser.add_argument('--w2v', type=int, default=0)
    parser.add_argument('--window_size', type=int, default=5)
    parser.add_argument('--w2v_ep', type=int, default=1)
    parser.add_argument('--w2v_lr', type=int, default=0.025)
    parser.add_argument('--min_count', type=int, default=5)
    # model arguments
    parser.add_argument('--doc_len', type=int, default=25)
    parser.add_argument('--sent_len', type=int, default=50)
    parser.add_argument('--attention', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--n_epoch', type=int, default=100)
    args = parser.parse_args()

    word_dict = dataSet.build_dict(
        fp='data/{}.train'.format(args.dataout),
        w_freq=5,
    )

    train_data = dataSet.data_generator(
        fp='./data/{}.train'.format(args.dataout),
        word_dict=word_dict,
        doc_len=args.doc_len,
        sent_len=args.sent_len,
        batch_size=args.batch_size
    )

    test_data = dataSet.data_generator(
        fp='./data/{}.test'.format(args.dataout),
        word_dict=word_dict,
        doc_len=args.doc_len,
        sent_len=args.sent_len,
        batch_size=args.batch_size
    )

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(graph=tf.Graph(), config=config) as sess:

        emb_pretrain = np.random.normal(size=[len(word_dict), args.emb_dim])
        emb_pretrain[:2] = 0

        model = MatchModel(
            n_word=len(word_dict),
            emb_dim=args.emb_dim,
            doc_len=args.doc_len,
            sent_len=args.sent_len,
            attention=args.attention,
            emb_pretrain=emb_pretrain,
        )
        predict = model.predict
        writer = tf.summary.FileWriter("board", sess.graph)

        loss = model.loss
        auc = model.auc

        optimizer = tf.train.AdamOptimizer()
        train_op = optimizer.minimize(loss)

        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        for epoch in range(args.n_epoch):
            epoch_loss, epoch_metric = [], []
            epoch_metric_outside = []
            for batch in tqdm(train_data, ncols=50):
                jds, cvs, labels = batch
                inf_masks, zero_masks = model.get_masks(jds, cvs)
                predict_data, loss_data, auc_data, _ = sess.run(
                    [predict, loss, auc, train_op],
                    feed_dict={
                        'jds:0': jds,
                        'cvs:0': cvs,
                        'inf_masks:0': inf_masks,
                        'zero_masks:0': zero_masks,
                        'labels:0': labels
                    })
                epoch_loss.append(loss_data)
                epoch_metric.append(auc_data)
                outside_auc_data = metrics.roc_auc_score(labels, predict_data)
                epoch_metric_outside.append(outside_auc_data)
            val_loss, val_metric = [], []
            val_metric_outside = []
            for batch in test_data:
                jds, cvs, labels = batch
                inf_masks, zero_masks = model.get_masks(jds, cvs)
                predict_data, loss_data, auc_data = sess.run(
                    [predict, loss, auc],
                    feed_dict={
                        'jds:0': jds,
                        'cvs:0': cvs,
                        'inf_masks:0': inf_masks,
                        'zero_masks:0': zero_masks,
                        'labels:0': labels,
                    })
                val_loss.append(loss_data)
                val_metric.append(auc_data)
                outside_auc_data = metrics.roc_auc_score(labels, predict_data)
                val_metric_outside.append(outside_auc_data)
            print(
                'epoch: {}\n'.format(epoch),
                'train loss: {:.3f} train metric: {:.3f} {:.3f}\n'.format(
                    np.array(epoch_loss).mean(),
                    np.array(epoch_metric).mean(),
                    np.array(epoch_metric_outside).mean()),
                'valid loss: {:.3f} valid metric: {:.3f} {:.3f}\n'.format(
                    np.array(val_loss).mean(),
                    np.array(val_metric).mean(),
                    np.array(val_metric_outside).mean()),
            )
    writer.close()
