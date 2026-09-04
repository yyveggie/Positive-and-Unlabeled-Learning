'''CSV file format'''

import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from sklearn.naive_bayes import GaussianNB
import random
import time
from Conversion import CSV

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
    print("iter", k)
    p, u, test_x, test_y = SplitData(k, texts_1, texts_0)
    p = pd.DataFrame(p)
    u = pd.DataFrame(u)
    i = 1
    W_0 = len(u)
    W_1 = len(u)
    while W_1 <= W_0:
        print("iter/stage:", k, "/", i)
        W_0 = W_1  # the previous W1 becomes the next W0
        print("stage %d"% i + " W0:", W_0)
        X_train = pd.concat([p, u])
        y_train = pd.Series([1] * len(p) + [0] * len(u))
        rf = GaussianNB()
        rf.fit(X_train, y_train)
        pred = rf.predict(u)
        W_1 = np.sum(pred == 1) # W1 is reset to the number predicted positive
        print("stage %d"% i + " W1:", W_1)
        u = pd.DataFrame([u.iloc[o] for o in range(len(pred)) if pred[o] == 0]) # keep the members of U predicted negative
        i = i + 1
        if W_1 == 0:
            break
    y_pred = rf.predict(test_x)
    f1_binary.append(f1_score(test_y, y_pred, average='binary'))
    accuracy.append(accuracy_score(test_y, y_pred))

print("F1-Score：", np.mean(np.array(f1_binary)))
print("Accuracy：", np.mean(np.array(accuracy)))