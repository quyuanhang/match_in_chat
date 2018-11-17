from tqdm import tqdm
import numpy as np


def data_generator(fp, word_dict, batch_size=0):
    data = []
    with open(fp) as f:
        for line in tqdm(f):
            jid, jd, gid, cv, label = line[:-1].split('\001')
            jd = doc_to_array(jd, word_dict)
            cv = doc_to_array(cv, word_dict)
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


def doc_to_array(raw_str, word_dict):
    doc = [sent_to_array(x, word_dict) for x in raw_str.split('\t')]
    return doc


def sent_to_array(raw_str, word_dict):
    sent = [word_dict.get(x, 0) for x in raw_str.split(' ')]
    return sent


def load_word_emb(filename):
    words = []
    embs = []
    start = True
    with open(filename) as f:
        for line in tqdm(f):
            if start:
                start = False
                continue
            data = line.strip().split(' ')
            word = data[0]
            emb = [float(x) for x in data[1:]]
            words.append(word)
            embs.append(emb)
    word_dict = {k: v for v, k in enumerate(words, 1)}
    dim = len(embs[0])
    embs.insert(0, [0]*dim)
    embs = np.array(embs)
    return word_dict, embs

