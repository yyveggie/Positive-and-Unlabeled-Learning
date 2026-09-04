'''CSV format'''

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import f1_score
import random
from Conversion import CSV

f1 = []
accuracy = []
path = r'data/mushroom.csv'

texts_1, texts_0 = CSV(path)

def SplitData(k, texts_1, texts_0):                                            # ten-fold cross-validation
    percent_p = 3                                                              # number of folds assigned to P
    texts_1 = np.array_split(texts_1, 10)                                      # split the class-1 samples into ten folds
    texts_0 = np.array_split(texts_0, 10)                                      # split the class-0 samples into ten folds
    test_x = list(texts_1[k]) + list(texts_0[k])                               # test x; one fold is taken each round
    test_y = list(len(texts_1[k]) * [1]) + list(len(texts_0[k]) * [0])         # test y; positive is 1, negative is 0

    index_rest = sorted(set(range(10)) - {k})                             # indices left after removing the test fold
    index_p = random.sample(index_rest, percent_p)                             # randomly choose percent_p indices for P
    texts_1 = np.array(texts_1)                                                # convert to an array so multi-dimensional indexing works
    p = texts_1[index_p]                                                       # the P set

    index_except_p = sorted(set(range(10)) - {k} - set(index_p))          # indices left after removing the test fold and P
    texts_0 = np.array(texts_0)                                                # convert to an array so multi-dimensional indexing works
    u = list(texts_1[index_except_p]) + list(texts_0[index_rest])              # the U set
    p = np.array([j for i in p for j in i])
    u = np.array([j for i in u for j in i])
    return p, u, test_x, test_y

for k in range(10):
    print("Iter：", k)
    p, u, test_x, test_y = SplitData(k, texts_1, texts_0)
    C = GaussianNB()
    C.fit(np.vstack((p, u)), [1] * p.shape[0] + [0] * u.shape[0])
    y_pred = C.predict(test_x)

    f1.append(f1_score(test_y, y_pred, average='binary'))
    accuracy.append(accuracy_score(test_y, y_pred))

print("F1-Score：", np.mean(np.array(f1)))
print("Accuracy：", np.mean(np.array(accuracy)))