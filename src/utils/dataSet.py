from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split


def data_generator(fp, word_dict, doc_len, sent_len, batch_size=0):
    data = []
    with open(fp) as f:
        for line in tqdm(f):
            jid, jd, gid, cv, label = line.strip().split('\001')
            jd = doc_to_array(jd, word_dict, doc_len, sent_len)
            cv = doc_to_array(cv, word_dict, doc_len, sent_len)
            label = int(label)
            data.append([jd, cv, label])
    if not batch_size:
        return list(zip(*data))
    data_batches = []
    for batch in range(len(data) // batch_size - 1):
        batch_data = data[batch*batch_size: (batch+1)*batch_size]
        batch_data = list(zip(*batch_data))
        data_batches.append(batch_data)
    return data_batches


def doc_to_array(raw_str, word_dict, doc_len, sent_len):
    doc = [sent_to_array(x, word_dict, sent_len) for x in raw_str.split('\t')]
    if len(doc) < doc_len:
        doc += [[0] * sent_len] * (doc_len - len(doc))
    else:
        doc = doc[:doc_len]
    doc = np.array(doc)
    return doc


def sent_to_array(raw_str, word_dict, sent_len):
    sent = [word_dict.get(x, 0) for x in raw_str.split(' ')]
    if len(sent) < sent_len:
        sent += [0] * (sent_len - len(sent))
    else:
        sent = sent[:sent_len]
    return sent


def load_word_emb(filename, emb_dim):
    words = []
    embs = []
    start = True
    with open(filename) as f:
        for line in tqdm(f):
            if start:
                start = False
                continue
            data = line.strip().split(' ')
            if len(data) != emb_dim + 1:
                continue
            word = data[0]
            emb = [float(x) for x in data[1:]]
            words.append(word)
            embs.append(emb)
    word_dict = {k: v for v, k in enumerate(words, 1)}
    dim = len(embs[0])
    embs.insert(0, [0.0]*dim)
    embs = np.array(embs)
    return word_dict, embs


def data_split(fpin, fpout, frac):
    with open('{}.positive'.format(fpin)) as f:
        posi_data = ['{}\0011'.format(x.strip()) for x in f]
    with open('{}.negative'.format(fpin)) as f:
        nega_data = ['{}\0010'.format(x.strip()) for x in f]
    data = posi_data + nega_data
    train, test = train_test_split(data, test_size=frac)
    with open('./data/{}.train'.format(fpout), 'w') as f:
        f.write('\n'.join(train))
    with open('./data/{}.test'.format(fpout), 'w') as f:
        f.write('\n'.join(test))
