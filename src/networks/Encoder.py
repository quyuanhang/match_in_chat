import torch
from torch import nn

class Encoder(nn.Module):
    def __init__(self, job=True):
        super(Encoder, self).__init__()
        self.total_word = vocab_size + 2
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(self.total_word, self.hidden_dim, padding_idx=0)
        self.job = job
        self.sen_maxl = sen_maxl
        self.doc_maxl = doc_maxl
        self.enc = Cnn()

    def forward(self, batch):
        batch_inputs, batch_len = self.batch_to_input(batch)
        batch_embedding = self.embedding(batch_inputs)

        batch_sen = self.enc(batch_embedding.unsqueeze(dim=1))
        return batch_sen, batch_len

    def batch_to_input(self, batch):
        batch_inputs = []
        batch_length = []
        for sample in batch:
            deta = 0
            len_sample = len(sample)

            batch_length.append(min(len_sample, self.doc_maxl))
            if len_sample > doc_maxl:
                sample = sample[:doc_maxl]
            else:
                deta = doc_maxl - len_sample
            for utterance in sample:
                batch_inputs.append(utterance)
            for i in range(deta):
                batch_inputs.append(np.zeros(self.sen_maxl))

        batch_inputs = Variable(torch.LongTensor(batch_inputs), requires_grad=False).cuda()
        return batch_inputs, batch_length


import tensorflow as tf


class Encoder:
    def __init__(self, n_word, emb_dim, doc_len, sent_len):
        pass

    def forward(self, x):
        pass

