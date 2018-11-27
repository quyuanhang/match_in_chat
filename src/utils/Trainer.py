import numpy as np
import tensorflow as tf
from networks import MatchModel
from tqdm import tqdm

from sklearn import metrics

def feed_dict(data, model: MatchModel):
    jds, cvs, labels = data
    inf_mask, zero_mask = model.get_masks(jds, cvs)
    fd = {
        'jd:0': jds,
        'cv:0': cvs,
        'inf_mask:0': inf_mask,
        'zero_mask:0': zero_mask,
        'label:0': labels
    }
    return fd


def train(
        sess: tf.Session,
        model: MatchModel,
        writer,
        train_generator,
        test_generator,
        lr=0.0005,
        n_epoch=100,
):

    predict = model.predict
    # loss, auc = model.loss_function()
    loss = model.loss_function()
    # optimizer = tf.train.AdamOptimizer(learning_rate=lr)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)
    train_op = optimizer.minimize(loss)
    merged = tf.summary.merge_all()
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    for epoch in range(n_epoch):
        # training
        losses = []
        aucs = []
        for i, batch in tqdm(list(enumerate(train_generator))):
            fd = feed_dict(batch, model)
            # sess.run(train_op, feed_dict=fd)
            # train_loss_data, auc_data = sess.run([loss, auc], feed_dict=fd)
            train_loss_data, predict_data, summery, _ = sess.run([loss, predict, merged, train_op], feed_dict=fd)
            writer.add_summary(summery, i)
            losses.append(train_loss_data)
            y = fd['label:0']
            fpr, tpr, _ = metrics.roc_curve(y, predict_data, pos_label=1)
            auc_data = metrics.auc(fpr, tpr)
            aucs.append(auc_data)
        print('epoch: {}, train loss: {}'.format(epoch, np.array(losses).mean()))
        print('epoch: {}, train auc: {}'.format(epoch, np.array(aucs).mean()))

        # predicts = []
        aucs = []
        for batch in tqdm(test_generator):
            fd = feed_dict(batch, model)
            predict_data = sess.run(predict, feed_dict=fd)
            # predicts.extend(list(predict_data.T[0]))
            y = fd['label:0']
            fpr, tpr, _ = metrics.roc_curve(y, predict_data, pos_label=1)
            auc_data = metrics.auc(fpr, tpr)
            aucs.append(auc_data)
        print('epoch: {},  valid auc: {}'.format(epoch, np.array(aucs).mean()))
