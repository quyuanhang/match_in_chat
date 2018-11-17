import tensorflow as tf
from networks import MatchModel
from tqdm import tqdm


def feed_dict(data, model: MatchModel):
    jds, cvs, labels = data
    inf_mask, zero_mask = model.get_masks(jds, cvs)
    fd = {
        model.jd: jds,
        model.cv: cvs,
        model.inf_mask: inf_mask,
        model.zero_mask: zero_mask,
        'label:0': labels
    }
    return fd


def train(
        sess: tf.Session,
        model: MatchModel,
        data_generator,
        test_data,
        lr=0.001,
        n_epoch=100):

    loss, auc = model.loss()
    optimizer = tf.train.AdamOptimizer(learning_rate=lr)
    train_op = optimizer.minimize(loss)
    for epoch in range(n_epoch):
        fd = feed_dict(test_data, model)
        auc_data = sess.run(auc, feed_dict=fd)
        print('epoch: {}, auc: {}'.format(epoch, auc_data))
        for batch in tqdm(data_generator):
            fd = feed_dict(batch, model)
            sess.run(train_op, feed_dict=fd)
        train_loss_data = sess.run(loss, feed_dict=fd)
        print('epoch: {}, train loss: {}'.format(epoch, train_loss_data))


