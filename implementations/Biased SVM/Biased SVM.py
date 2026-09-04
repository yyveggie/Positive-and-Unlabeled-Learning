'''Dataset is in LIBSVM format'''
'''Uses SVMLight'''

import os
import numpy as np
from Conversion import SVMlight
import random


path = r'data/mushroom.libsvm'
texts_1, texts_0 = SVMlight(path)
percent_p = 3  # number of folds assigned to P
texts_1 = np.array_split(texts_1, 10)  # split the class-1 samples into ten folds
texts_0 = np.array_split(texts_0, 10)  # split the class-0 samples into ten folds

for k in range(10):
    test = list(texts_1[k]) + list(texts_0[k])  # test x; one fold is taken each round
    test_y = list(len(texts_1[k]) * [1]) + list(len(texts_0[k]) * [-1])  # test y; positive is 1, negative is 0

    index_rest = sorted(set(range(10)) - {k})  # indices left after removing the test fold
    index_p = random.sample(index_rest, percent_p)  # randomly choose percent_p indices for P
    texts_1 = np.array(texts_1)  # convert to an array so multi-dimensional indexing works
    p = texts_1[index_p]  # the P set

    index_except_p = sorted(set(range(10)) - {k} - set(index_p))  # indices left after removing the test fold and P
    texts_0 = np.array(texts_0)  # convert to an array so multi-dimensional indexing works
    u = list(texts_1[index_except_p]) + list(texts_0[index_rest])  # the U set

    p = np.array([j for i in p for j in i])
    u = np.array([j for i in u for j in i])
    train = np.concatenate((p, u), axis=0)
    y = [1] * len(p) + [-1] * len(u)

    with open(r'D:\SVM_Light\svm_light_windows64\example1\train_.dat', 'w') as f:
        for i in range(len(y)):
            f.write(str(y[i]))
            f.write(' ')
            f.write(str(train[i]).replace(' ', '').replace('{', '').replace('}', '').replace(',', ' '))
            f.write('\n')
    with open(r'D:\SVM_Light\svm_light_windows64\example1\test_.dat', 'w') as f:
        for i in range(len(test_y)):
            f.write(str(test_y[i]))
            f.write(' ')
            f.write(str(test[i]).replace(' ', '').replace('{', '').replace('}', '').replace(',', ' '))
            f.write('\n')
    result = os.system('cd /d D:\SVM_Light\svm_light_windows64 && svm_learn.exe -z c -c 0.01 -j 10 example1/train_.dat example1/model && svm_classify.exe example1/test_.dat example1/model example1/predicidons')
    print(result)