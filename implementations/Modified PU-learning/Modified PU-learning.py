''' CSV file format '''

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import f1_score
import pandas as pd
from Conversion import CSV
import random

f1_binary = []
accuracy = []
path = r'data/mushroom.csv'

texts_1, texts_0 = CSV(path)

def SplitData(k, texts_1, texts_0):
    percent_p = 3  # number of folds assigned to P
    texts_1 = np.array_split(texts_1, 10)  # split the class-1 samples into ten folds
    texts_0 = np.array_split(texts_0, 10)  # split the class-0 samples into ten folds
    test_x = list(texts_1[k]) + list(texts_0[k])  # test x; one fold is taken each round
    test_y = list(len(texts_1[k]) * [1]) + list(len(texts_0[k]) * [0])  # test y; positive is 1, negative is 0

    index_rest = sorted(set(range(10)) - set([k]))  # indices left after removing the test fold
    index_p = random.sample(index_rest, percent_p)  # randomly choose percent_p indices for P
    texts_1 = np.array(texts_1)  # convert to an array so multi-dimensional indexing works
    p = texts_1[index_p]  # the P set

    index_except_p = sorted(set(range(10)) - set([k]) - set(index_p))  # indices left after removing the test fold and P
    texts_0 = np.array(texts_0)  # convert to an array so multi-dimensional indexing works
    u = list(texts_1[index_except_p]) + list(texts_0[index_rest])  # the U set
    p = np.array([j for i in p for j in i])
    u = np.array([j for i in u for j in i])
    return p, u, test_x, test_y

for k in range(10):
    p, u, test_x, test_y = SplitData(k, texts_1, texts_0)
    p = pd.DataFrame(p)
    u = pd.DataFrame(u)
    C = BernoulliNB()
    C.fit(pd.concat([p, u]), pd.Series([1] * p.shape[0] + [0] * u.shape[0]))
    lis = list(C.predict(u))
    Q = [u.iloc[i] for i in range(len(lis)) if lis[i] == 0]  # Q holds the samples predicted negative
    Q = pd.DataFrame(Q)
    RN = Q  # Q becomes the initial RN
    print("size of the initial RN:", len(RN))
    Q_ = Q  # set the two equal so the loop does not exit immediately
    j = 1  # iteration counter
    while len(Q) <= len(Q_) and p.shape[0] < len(RN):  # loop while the current negative count is at most the previous one and |P| is below |RN|
        print("iter:", k, j)
        Q_ = Q  # the current Q becomes the previous Q
        clf = BernoulliNB()
        clf.fit(pd.concat([p, RN]), pd.Series([1] * p.shape[0] + [0] * RN.shape[0]))  # train a classifier on P and the current RN
        lis = list(clf.predict(RN))  # score the current RN with that classifier
        Q = [RN.iloc[o] for o in range(len(lis)) if lis[o] == 0]  # keep the ones predicted negative
        Q = pd.DataFrame(Q)  # the new Q
        RN = Q
        print("size of the current RN:", len(RN))
        j += 1
        if len(Q) == len(Q_):
            break
    y_pred = clf.predict(test_x)  # once the stopping rule fires, return the last classifier and score the test set
    f1_binary.append(f1_score(test_y, y_pred, average='binary'))
    accuracy.append(accuracy_score(test_y, y_pred))

print("F1-Score：", np.mean(np.array(f1_binary)))
print("Accuracy：", np.mean(np.array(accuracy)))