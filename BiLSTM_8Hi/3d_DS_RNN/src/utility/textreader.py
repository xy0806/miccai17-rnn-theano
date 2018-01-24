import numpy as np
import nltk
import itertools

__author__ = 'uyaseen'


def tokenize(text):
    ll = [[nltk.word_tokenize(w), ' '] for w in text.split()]
    return list(itertools.chain(*list(itertools.chain(*ll))))


def get_one_hot(idx, vocab_size):
        arr = np.zeros(vocab_size)
        arr[idx] = 1
        return arr


def get_one_hot_vocab_list(vocab, vocab_size):
        l = []
        for j in vocab:
            l.append(get_one_hot(j, vocab_size))
        return l


def read_word_data(f_name, seq_length):
    data = tokenize(open(f_name, 'r').read())
    voc = list(set(data))
    print('data size: %i, vocab size: %i' % (len(data), len(voc)))
    words_to_ix = {wd: i for i, wd in enumerate(voc)}
    ix_to_words = {i: wd for i, wd in enumerate(voc)}
    x_ = [words_to_ix[wd] for wd in data]
    y_ = x_[1:] + x_[:1]
    x = []
    y = []
    e_idx = seq_length
    for i in xrange(0, len(x_) - 1, seq_length):
        if len(x_[i:e_idx]) == seq_length:
            x.append(x_[i:e_idx])
            y.append(y_[i:e_idx])
            e_idx += seq_length
    return [(x, y), (voc, ix_to_words, words_to_ix)]


def read_char_data(f_name, seq_length):
    data = open(f_name, 'r').read()
    voc = list(set(data))
    print('data size: %i, vocab size: %i' % (len(data), len(voc)))
    chars_to_ix = {ch: i for i, ch in enumerate(voc)}
    ix_to_chars = {i: ch for i, ch in enumerate(voc)}
    x_ = [chars_to_ix[ch] for ch in data]
    y_ = x_[1:] + x_[:1]
    x = []
    y = []
    e_idx = seq_length
    for i in xrange(0, len(x_) - 1, seq_length):
        if len(x_[i:e_idx]) == seq_length:
            x.append(x_[i:e_idx])
            y.append(y_[i:e_idx])
            e_idx += seq_length
    return [(x, y), (voc, ix_to_chars, chars_to_ix)]
