import numpy as np
import tensorflow as tf
from networks.Model import MatchModel
from tqdm import tqdm

from sklearn import metrics

def feed_dict(data, model: MatchModel):
    jds, cvs, labels = data
    inf_masks, zero_masks = model.get_masks(jds, cvs)
    fd = {
        'jds:0': jds,
        'cvs:0': cvs,
        'inf_masks:0': inf_masks,
        'zero_masks:0': zero_masks,
        'labels:0': labels
    }
    return fd


def train(
        sess: tf.Session,
        model: MatchModel,
        writer,
        train_data,
        test_data,
        lr=0.0005,
        n_epoch=100,
):

        predict = model.predict

        loss = model.loss

        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        train_op = optimizer.minimize(loss)

        sess.run(tf.global_variables_initializer())

        for epoch in range(n_epoch):
            epoch_loss = []
            epoch_metric_outside = []
            for batch in tqdm(train_data, ncols=50):
                fd = feed_dict(batch, model)
                predict_data, loss_data, _ = sess.run(
                    [predict, loss, train_op],
                    feed_dict=fd)
                epoch_loss.append(loss_data)
                outside_auc_data = metrics.roc_auc_score(fd['labels:0'], predict_data)
                epoch_metric_outside.append(outside_auc_data)
            val_loss = []
            val_metric_outside = []
            for batch in test_data:
                fd = feed_dict(batch, model)
                predict_data, loss_data = sess.run(
                    [predict, loss],
                    feed_dict=fd)
                val_loss.append(loss_data)
                outside_auc_data = metrics.roc_auc_score(fd['labels:0'], predict_data)
                val_metric_outside.append(outside_auc_data)
            print(
                'epoch: {}\n'.format(epoch),
                'train loss: {:.3f} train metric: {:.3f}\n'.format(
                    np.array(epoch_loss).mean(),
                    np.array(epoch_metric_outside).mean()),
                'valid loss: {:.3f} valid metric: {:.3f}\n'.format(
                    np.array(val_loss).mean(),
                    np.array(val_metric_outside).mean()),
            )

