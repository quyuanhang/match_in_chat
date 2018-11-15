import torch
from torch import nn


class Classfier(nn.Module):
    def __init__(self):
        super(Classfier, self).__init__()
        self.hidden_dim = hidden_dim
        self.Bil = nn.Bilinear(in1_features=2 * self.hidden_dim, in2_features=2 * self.hidden_dim, out_features=self.hidden_dim)
        self.MLP = nn.Sequential(nn.Sigmoid(),
                                 nn.Linear(in_features=hidden_dim, out_features=1),
                                 nn.Sigmoid())
    def forward(self, geek, job):
        mix = self.Bil(geek, job)
        match_score = self.MLP(mix)
        return match_score


import tensorflow as tf


class BiMLP:
    def __init__(self, emb_dim):
        self.emb_dim = emb_dim
        self.bilinear_weights = tf.Variable(tf.random_normal(
            shape=[emb_dim, 2*emb_dim, 2*emb_dim]
        ))

    def forward(self, jd_data, cv_data):
        pass

    def bilinear(self, jd_data, cv_data, weights):
        x = tf.matmul(jd_data, weights)
        x = tf.multiply(tmp, cv_data)
        x = tf.reduce_sum(x, axis=1)



