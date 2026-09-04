'''CSV file format'''

from Conversion import CSV
import numpy as np
import random
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import pandas as pd

f1_binary = []
accuracy = []
T = 4.8  # threshold
k = 5

path = r'data/mushroom.csv'

texts_1, texts_0 = CSV(path)

def SplitData(t, texts_1, texts_0):  # ten-fold cross-validation
    percent_p = 3  # number of folds assigned to P
    texts_1 = np.array_split(texts_1, 10)  # split the class-1 samples into ten folds
    texts_0 = np.array_split(texts_0, 10)  # split the class-0 samples into ten folds
    test_x = list(texts_1[t]) + list(texts_0[t])  # test x; one fold is taken each round
    test_y = list(len(texts_1[t]) * [1]) + list(len(texts_0[t]) * [0])  # test y; positive is 1, negative is 0

    index_rest = sorted(set(range(10)) - set([t]))  # indices left after removing the test fold
    index_p = random.sample(index_rest, percent_p)  # randomly choose percent_p indices for P
    texts_1 = np.array(texts_1)                                                # convert to an array so multi-dimensional indexing works
    p = texts_1[index_p]                                                       # the P set

    index_except_p = sorted(set(range(10)) - set([t]) - set(index_p))          # indices left after removing the test fold and P
    texts_0 = np.array(texts_0)                                                # convert to an array so multi-dimensional indexing works
    u = list(texts_1[index_except_p]) + list(texts_0[index_rest])              # the U set
    p = np.array([j for i in p for j in i])
    u = np.array([j for i in u for j in i])
    return p, u, test_x, test_y

for o in range(10):
    print("Iter：", o, '\n', '*' * 50)
    RN = []
    R_ = []
    p, u, test_x, test_y = SplitData(o, texts_1, texts_0)
    for i in range(len(u)):
        sim = {}
        sum_sim = []
        ui = u[i]
        for j in range(len(p)):
            pj = p[j]
            cos = np.dot(ui, pj) / (np.linalg.norm(ui) * (np.linalg.norm(pj)))  # cosine distance
            sim[j] = cos
        sim = sorted(sim.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)  # sort the dictionary by cosine value
        for n in range(k):  # take the indices of the k largest
            sum_sim.append(sim[n][1])
        value = sum(sum_sim)   # sum the top-k similarities
        wi = value - T

        if wi < 0:
            RN.append(u[i])
        else:
            R_.append(u[i])
    p = pd.DataFrame(p)
    u = pd.DataFrame(u)
    test_x = pd.DataFrame(test_x)
    test_y = pd.DataFrame(test_y)
    RN = pd.DataFrame(RN)
    R_ = pd.DataFrame(R_)

    x_train = pd.concat([p, RN])
    y_train = pd.Series(list([1] * len(p)) + list([0] * len(RN)))

    svc = SVC(kernel='linear')
    svc.fit(x_train, y_train)
    lis = list(svc.predict(R_))
    W = pd.DataFrame([R_.iloc[o] for o in range(len(lis)) if lis[o] == 0])  # keep the ones predicted negative
    Q = pd.DataFrame([R_.iloc[o] for o in range(len(lis)) if lis[o] == 1])
    while len(W) > 0:
        print("len(W):", len(W) )
        R = Q
        RN = pd.concat([RN, W])
        clf = SVC(kernel='linear')
        X = np.vstack((p, RN))
        y = pd.Series(list([1] * len(p)) + list([0] * len(RN)))
        clf.fit(X, y)
        llis = list(clf.predict(R))
        W = pd.DataFrame([R.iloc[o] for o in range(len(llis)) if llis[o] == 0])
        Q = pd.DataFrame([R.iloc[o] for o in range(len(llis)) if llis[o] == 1])

    y_pred = pd.DataFrame(clf.predict(test_x))  # once the stopping rule fires, return the last classifier and score the test set
    f1_binary.append(f1_score(test_y, y_pred, average='binary'))
    accuracy.append(accuracy_score(test_y, y_pred))
    print(f1_score(test_y, y_pred, average='binary'))
    print(accuracy_score(test_y, y_pred))

print("F1-Score：", np.mean(np.array(f1_binary)))
print("Accuracy：", np.mean(np.array(accuracy)))