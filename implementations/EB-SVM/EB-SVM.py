from sklearn.svm import SVC
import numpy as np
import random
from Conversion import CSV
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import pandas as pd

svc = SVC(C=1.0, kernel='rbf', gamma='auto', probability=True, random_state=2018)

f1 = []
accuracy = []
path = r'data/mushroom.csv'
C_p = 0.05
C_n = 0.3

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
    r = len([j for i in texts_1[index_except_p] for j in i])  # number of positives in U
    p = np.array([j for i in p for j in i])
    u = np.array([j for i in u for j in i])
    return p, u, test_x, test_y, r


def biased_svm(cost_fp, cost_fn, P, U, test_x, svm=svc, return_proba=True):
    assert cost_fn > cost_fp > 0, 'a false negative must cost more than a false positive'
    X_train = np.r_[P, U]
    y_train = np.r_[np.ones(len(P)), np.zeros(len(U))]
    weight = [cost_fn if i else cost_fp for i in y_train]
    svm.fit(X_train, y_train, sample_weight=weight)
    y_pred = svm.predict(test_x)
    if return_proba:
        y_prob = svm.predict_proba(U)[:, -1]
        return y_pred, y_prob, svm
    else:
        return y_pred, svm


for k in range(10):
    print("Iter：", k)
    p, u, test_x, test_y, r = SplitData(k, texts_1, texts_0)
    while True:
        y_pred, y_prob, svm = biased_svm(C_p, C_n, p, u, u)
        lis = list(y_pred)
        S = pd.DataFrame([pd.DataFrame(u).iloc[o] for o in range(len(lis)) if lis[o] == 1])
        H = pd.DataFrame([pd.DataFrame(u).iloc[o] for o in range(len(lis)) if lis[o] == 0])
        S_score = pd.DataFrame([pd.DataFrame(y_prob).iloc[o] for o in range(len(lis)) if lis[o] == 1])
        t = len(S)
        if r == 0:
            break
        elif 0 < t < r:
            CP = np.array(S)
            _u = H
            r = r - t
        elif t > r:
            _S = S.astype(np.float64)
            c = np.insert(_S, 0, values=S_score, axis=1)
            CP = np.array(S[np.argsort(c[:, 0])][:r])
            _u = np.r_[np.array(H), np.array(S[np.argsort(c[:, 0])][r:])]
            r = 0
        else:
            break
        p = np.r_[p, CP]
        u = _u

    pred = svm.predict(test_x)
    f1.append(f1_score(test_y, pred, average='binary'))
    accuracy.append(accuracy_score(test_y, pred))
    print(f1)
    print(accuracy)

print("Mean of F1-Score：", np.mean(np.array(f1)))
print("Mean of Accuracy：", np.mean(np.array(accuracy)))