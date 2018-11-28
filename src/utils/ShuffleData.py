import sys
import numpy as np

def load_data(fp, apd=None):
    with open(fp) as f:
        data = f.read().strip().split('\n')
    data = [x.strip().split('\001') for x in data]
    if apd:
        for d in data:
            d.append(apd)
    return data

def shuffle(data):
    n = len(data)
    permutation = np.random.permutation(n)
    data = [data[i] for i in permutation]
    return data

def save_data(fp, data):
    with open(fp, 'w') as f:
        for d in data:
            f.write('\001'.join(d))
            f.write('\n')
    return

if __name__ == '__main__':
    train_neg = load_data('../Data/interview_split/train/interview_split.negative', '0')
    train_posi = load_data('../Data/interview_split/train/interview_split.positive', '1')
    train = shuffle(train_neg + train_posi)

    test_neg = load_data('../Data/interview_split/test/interview_split.negative', '0')
    test_posi = load_data('../Data/interview_split/test/interview_split.positive', '1')
    test = shuffle(test_neg + test_posi)

    train_pair = set([(d[0], d[3]) for d in train])
    test = [d for d in test if (d[0], d[3]) not in train_pair]

    save_data('data/interview.train', train)
    save_data('data/interview.test', test)



