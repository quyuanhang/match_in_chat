import tensorflow as tf
from tensorflow import keras
from keras.datasets import imdb
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import sys
sys.path.append('../src/')
from sklearn import metrics

n_word = 10000
emb_dim = 100
n_epoch = 100
batch_size = 128
doc_len = 256

def pad(data):
    data = keras.preprocessing.sequence.pad_sequences(data,
                                                      value=0,
                                                      padding='post',
                                                      maxlen=doc_len)
    return data


def build_graph():
    x = tf.placeholder(dtype=tf.int32, shape=[None, doc_len], name='x')

    embedding = keras.layers.Embedding(
        input_dim=n_word, output_dim=emb_dim, embeddings_initializer='RandomNormal', mask_zero=True)
    x = embedding(x)
    x = tf.reshape(x, shape=[-1, doc_len, emb_dim, 1])

    x = keras.layers.ZeroPadding2D(padding=(2, 0))(x)
    x = keras.layers.Conv2D(filters=emb_dim, kernel_size=[5, emb_dim])(x)

    x = tf.transpose(x, [0,1,3,2])
    x = keras.layers.ZeroPadding2D(padding=(1, 0))(x)
    x = keras.layers.Conv2D(filters=emb_dim, kernel_size=[3, emb_dim])(x)

    x = tf.reshape(x, shape=[-1, doc_len, emb_dim])
    x = tf.reduce_mean(x, axis=1)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.Dense(1, activation='sigmoid')(x)
    return x

def build_loss(predict):
    predict = tf.squeeze(predict)
    label = tf.placeholder(dtype=tf.int32, shape=[None], name='y')
    loss = tf.losses.log_loss(label, predict)
    return loss

if __name__ == '__main__':


    (train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=n_word)
    train_data = pad(train_data)
    test_data = pad(test_data)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(graph=tf.Graph(), config=config) as sess:

        # model = keras.Sequential()
        # model.add(keras.layers.Embedding(n_word, 16))
        # model.add(keras.layers.GlobalAveragePooling1D())
        # model.add(keras.layers.Dense(16, activation=tf.nn.relu))
        # model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))
        # model.add(keras.layers.Lambda(g))

        # model.summary()

        # model.compile(optimizer=tf.train.AdamOptimizer(),
        #               loss='binary_crossentropy',
        #               metrics=['accuracy'])

        x_val = train_data[:1000]
        partial_x_train = train_data[10000:]

        y_val = train_labels[:1000]
        partial_y_train = train_labels[10000:]

        # history = model.fit(partial_x_train,
        #                     partial_y_train,
        #                     epochs=40,
        #                     batch_size=512,
        #                     validation_data=(x_val, y_val),
        #                     verbose=1)

        # results = model.evaluate(test_data, test_labels)

        # print(results)

        g = build_graph()
        loss = build_loss(g)

        optimizer = tf.train.AdamOptimizer()
        train_op = optimizer.minimize(loss)

        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        for epoch in range(n_epoch):
            epoch_loss, epoch_metric = [], []
            for i in range(len(partial_x_train) // batch_size):
                batch_x = partial_x_train[i*batch_size: (i+1)*batch_size]
                batch_y = partial_y_train[i*batch_size: (i+1)*batch_size]
                loss_data, pre_data, _ = sess.run([loss, g, train_op], feed_dict={'x:0': batch_x, 'y:0': batch_y})
                auc = metrics.roc_auc_score(batch_y, pre_data)
                epoch_loss.append(loss_data)
                epoch_metric.append(auc)
            val_pre = sess.run(g, feed_dict={'x:0': x_val})
            val_metric = metrics.roc_auc_score(y_val, val_pre)
            print('epoch: {}, train loss: {:.3f}, train metric: {:.3f}, valid metric: {:.3f}'.format(
                epoch, np.array(epoch_loss).mean(), np.array(epoch_metric).mean(), val_metric))












