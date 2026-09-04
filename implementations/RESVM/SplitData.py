'''LIBSVM data format'''

import numpy as np
import random

# this script splits the dataset into ten folds and writes them into the ensemblesvm directory; ensemblesvm is then run through Cygwin to train and predict

from Conversion import Libsvm

path = r'data/mushroom.libsvm'
percent_p = 3  # number of folds assigned to P
texts_1, texts_0 = Libsvm(path)
texts_1 = np.array_split(texts_1, 10)  # split the class-1 samples into ten folds
texts_0 = np.array_split(texts_0, 10)  # split the class-0 samples into ten folds

for k in range(10):
    test = list(texts_1[k]) + list(texts_0[k])  # test x; one fold is taken each round
    test_y = list(len(texts_1[k]) * [1]) + list(len(texts_0[k]) * [-1])  # test y; positive is 1, negative is 0

    index_rest = sorted(set(range(10)) - set([k]))  # indices left after removing the test fold
    index_p = random.sample(index_rest, percent_p)  # randomly choose percent_p indices for P
    texts_1 = np.array(texts_1)  # convert to an array so multi-dimensional indexing works
    p = texts_1[index_p]  # the P set

    index_except_p = sorted(set(range(10)) - set([k]) - set(index_p))  # indices left after removing the test fold and P
    texts_0 = np.array(texts_0)  # convert to an array so multi-dimensional indexing works
    u = list(texts_1[index_except_p]) + list(texts_0[index_rest])  # the U set

    p = np.array([j for i in p for j in i])
    u = np.array([j for i in u for j in i])
    X = np.concatenate((p, u), axis=0)
    y = [1] * len(p) + [-1] * len(u)

    with open(r'data/train_{}.libsvm'.format(str(k)), 'w') as f:
        for i in range(len(X)):
            f.write(str(y[i]))
            f.write(' ')
            f.write(str(X[i]).replace(' ', '').replace('{', '').replace('}', '').replace(',', ' '))
            f.write('\n')
    with open(r'data/test_{}.libsvm'.format(str(k)), 'w') as f:
        for o in range(len(test)):
            f.write(str(test_y[o]))
            f.write(' ')
            f.write(str(test[o]).replace(' ', '').replace('{', '').replace('}', '').replace(',', ' '))
            f.write('\n')

## then call ensemblesvm through Cygwin to train and predict

## finally run Performance.py to compute the F1 score and accuracy