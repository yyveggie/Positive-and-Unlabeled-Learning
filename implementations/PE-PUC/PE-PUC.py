"""CSV format"""

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import f1_score
import random
from Conversion import CSV
import pandas as pd

f1 = []
accuracy = []
path = r'data/mushroom.csv'
alpha = 0.99
l = 10
sigma = 1.0
_lambda = 0.7

texts_1, texts_0 = CSV(path)


def SplitData(k, texts_1, texts_0):  # ten-fold cross-validation
    percent_p = 3  # number of folds assigned to P
    texts_1 = np.array_split(texts_1, 10)  # split the class-1 samples into ten folds
    texts_0 = np.array_split(texts_0, 10)  # split the class-0 samples into ten folds
    test_x = list(texts_1[k]) + list(texts_0[k])  # test x; one fold is taken each round
    test_y = list(len(texts_1[k]) * [1]) + list(len(texts_0[k]) * [0])  # test y; positive is 1, negative is 0

    index_rest = sorted(set(range(10)) - {k})  # indices left after removing the test fold
    index_p = random.sample(index_rest, percent_p)  # randomly choose percent_p indices for P
    texts_1 = np.array(texts_1)  # convert to an array so multi-dimensional indexing works
    p = texts_1[index_p]  # the P set

    index_except_p = sorted(set(range(10)) - {k} - set(index_p))  # indices left after removing the test fold and P
    texts_0 = np.array(texts_0)  # convert to an array so multi-dimensional indexing works
    u = list(texts_1[index_except_p]) + list(texts_0[index_rest])  # the U set
    p = np.array([j for i in p for j in i])
    u = np.array([j for i in u for j in i])
    return p, u, test_x, test_y


for k in range(10):
    print("Iter：", k)
    p, u, test_x, test_y = SplitData(k, texts_1, texts_0)
    ### Extract RN
    nb = GaussianNB()
    nb.fit(np.vstack((p, u)), [1] * p.shape[0] + [0] * u.shape[0])
    y_pred = nb.predict(u)
    RN = u[y_pred == 0]
    Q = u[y_pred == 1]  # U-RN
    ### Enlarge P
    n = len(Q)
    print(n)
    W = np.zeros((l + n, l + n))
    PL = p[np.random.choice(len(p), size=l, replace=False)]
    X = np.vstack((PL, Q))
    for i in range(l + n):
        for j in range(l + n):
            if i == j:
                W[i][j] = 0
            elif i < j:
                W[i][j] = np.exp(-(np.linalg.norm(X[i] - X[j])) ** 2 / 2 * sigma ** 2)
            elif i > j:
                W[i][j] = W[j][i]
    D = np.zeros(W.shape)
    for z in range(W.shape[0]):
        D[z][z] = np.sum(W[z])
    # S = np.sqrt(D) * W * np.sqrt(D)    # normalization as printed drives every matrix entry to 0, so it is not applied here
    y = np.array([[1] * l + [1] * n]).T
    # f = (1-alpha)*((1-alpha*W)**(-1))*y   # the y of the original makes the result unsortable
    f = (1 - alpha) * ((1 - alpha * W) ** (-1)) * y
    _length = np.ceil(_lambda * n).astype(int)
    f = f[l:]
    _f = np.zeros(f.shape[0])
    for i in range(f.shape[0]):
        _f[i] = np.sum(f[i])
    _Q = Q.astype(np.float64)
    c = np.insert(_Q, 0, values=_f, axis=1)    # insert _f as the first column of _Q
    RP = Q[np.argsort(c[:, 0])][:_length]   # sort by the first column and select
    RN_2 = Q[np.argsort(c[:, 0])][_length:]
    ### Extract RN
    _p = np.vstack((p, RP))
    _u = np.vstack((RN, RN_2))
    clf = GaussianNB()
    clf.fit(np.vstack((_p, _u)), np.array([1] * len(_p) + [0] * len(_u)))
    lis = list(clf.predict(_u))
    _u = pd.DataFrame(_u)
    _RN = np.array([_u.iloc[o] for o in range(len(lis)) if lis[o] == 0])
    ### Build Classifier
    _clf = GaussianNB()
    _clf.fit(np.vstack((_p, _RN)), np.array([1] * len(_p) + [0] * len(_RN)))
    y_pred = _clf.predict(test_x)
    f1.append(f1_score(test_y, y_pred, average='binary'))
    accuracy.append(accuracy_score(test_y, y_pred))
    print(f1)
    print(accuracy)

print("F1-Score：", np.mean(np.array(f1)))
print("Accuracy：", np.mean(np.array(accuracy)))
