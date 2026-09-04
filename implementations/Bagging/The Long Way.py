'''Dataset is in LIBSVM format'''

from __future__ import division, print_function
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from Conversion import Libsvm
import random
from svmutil import *
import time

''' Create data '''

s = time.time()
path = r'data/mushroom.libsvm'
f1 = []
accuracy = []

texts_1, texts_0 = Libsvm(path)

def Create_data(k, texts_1, texts_0):
    percent_p = 3                                                           # number of folds assigned to P
    texts_1 = np.array_split(texts_1, 10)                                   # split the class-1 samples into ten folds
    texts_0 = np.array_split(texts_0, 10)                                   # split the class-0 samples into ten folds
    test_x = list(texts_1[k]) + list(texts_0[k])                            # test x; one fold is taken each round
    test_y = list(len(texts_1[k]) * [1]) + list(len(texts_0[k]) * [0])      # test y; positive is 1, negative is 0

    index_rest = sorted(set(range(10)) - set([k]))                          # indices left after removing the test fold
    index_p = random.sample(index_rest, percent_p)                          # randomly choose percent_p indices for P
    texts_1 = np.array(texts_1)                                             # convert to an array so multi-dimensional indexing works
    p = texts_1[index_p]                                                    # the P set

    index_except_p = sorted(set(range(10)) - set([k]) - set(index_p))       # indices left after removing the test fold and P
    texts_0 = np.array(texts_0)                                             # convert to an array so multi-dimensional indexing works
    u = list(texts_1[index_except_p]) + list(texts_0[index_rest])           # the U set
    return p, u, test_x, test_y

''' Transductive PU learning '''

for k in range(10):                                        # ten-fold cross-validation
    p, u, test_x, test_y = Create_data(k, texts_1, texts_0)
    NP = sum([len(i) for i in p])                          # size of P
    NU = sum([len(i) for i in u])                          # size of U
    shape = len(test_y)                                    # size of the test set

    T = 20                                      # number of iterations
    K = NP                                      # number of samples bootstrapped from U each round
    train_label = np.zeros(shape=(NP + K, ))    # training labels start at 0 (training set = P + U)
    train_label[:NP] = 1.0                      # the first NP entries are P and get label 1

    n_oob = np.zeros(shape=(shape, ))
    f_oob = np.zeros(shape=(shape, 2))

    p = np.array([j for i in p for j in i])
    u = np.array([j for i in u for j in i])
    test_x = np.array(test_x)
    test_y = np.array(test_y)

    for i in range(T):
        print("Iter：", i)
        # indices of the bootstrap resample from U
        bootstrap_sample = np.random.choice(np.arange(NU), replace=True, size=K)
        # training set = P plus the resample from U
        data_bootstrap = np.concatenate((p, u[bootstrap_sample]), axis=0)
        # Train model
        model = svm_train([1] * len(p) + [0] * len(bootstrap_sample), data_bootstrap, '-b 1')
        # score the test set and accumulate the predictions
        idx_alltest = sorted(set(range(len(test_y))))
        p_labels, p_acc, p_vals = svm_predict(test_y[idx_alltest], test_x[idx_alltest], model, '-b 1')
        f_oob[idx_alltest] += np.array(p_vals)[idx_alltest]
        n_oob[idx_alltest] += 1
    predict_proba = f_oob[:, 0] / n_oob         # average the second column, the probability of belonging to P

    y_pred = np.zeros(len(test_y))
    y_true = test_y
    for i in range(len(predict_proba)):
        if predict_proba[i] >= 0.5:
            y_pred[i] = 1

    f1.append(f1_score(y_true, y_pred, average='binary'))
    accuracy.append(accuracy_score(y_true, y_pred))

print("F1-Score：", np.mean(np.array(f1)))
print("Accuracy：", np.mean(np.array(accuracy)))
print("Time：", time.time() - s)