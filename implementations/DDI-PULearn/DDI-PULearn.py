'''CSV file format'''

from Conversion import CSV
from sklearn.svm import SVC
import numpy as np
import random

f1 = []
accuracy = []
path = r'data/mushroom.csv'

k = 5   # k nearest neighbours
T = 4.8  # threshold

texts_1, texts_0 = CSV(path)

def SplitData(k, texts_1, texts_0):                                            # ten-fold cross-validation
    percent_p = 3                                                              # number of folds assigned to P
    texts_1 = np.array_split(texts_1, 10)                                      # split the class-1 samples into ten folds
    texts_0 = np.array_split(texts_0, 10)                                      # split the class-0 samples into ten folds
    test_x = list(texts_1[k]) + list(texts_0[k])                               # test x; one fold is taken each round
    test_y = list(len(texts_1[k]) * [1]) + list(len(texts_0[k]) * [0])         # test y; positive is 1, negative is 0

    index_rest = sorted(set(range(10)) - set([k]))                             # indices left after removing the test fold
    index_p = random.sample(index_rest, percent_p)                             # randomly choose percent_p indices for P
    texts_1 = np.array(texts_1)                                                # convert to an array so multi-dimensional indexing works
    p = texts_1[index_p]                                                       # the P set

    index_except_p = sorted(set(range(10)) - set([k]) - set(index_p))          # indices left after removing the test fold and P
    texts_0 = np.array(texts_0)                                                # convert to an array so multi-dimensional indexing works
    u = list(texts_1[index_except_p]) + list(texts_0[index_rest])              # the U set
    p = np.array([j for i in p for j in i])
    u = np.array([j for i in u for j in i])
    return p, u, test_x, test_y

for k in range(10):
    print("Iter：", k, '\n', '*' * 50)
    p, u, test_x, test_y = SplitData(k, texts_1, texts_0)
    RN = []
    RU = []
    V = []
    for i in range(len(u)):
        sim = {}
        sum_sim = []
        ui = u[i]
        for j in range(len(p)):
            pj = p[j]
            cos = np.dot(ui, pj) / (np.linalg.norm(ui) * (np.linalg.norm(pj)))  # cosine distance
            sim[i] = cos
        sim = sorted(sim.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)  # sort the dictionary by cosine value
        for n in range(k):  # take the indices of the k largest
            sum_sim.append(sim[n][1])
        value = sum(sum_sim)  # sum the top-k similarities
        wi = value - T

        if wi < 0:
            RN.append(u[i])
        else:
            RU.append(u[i])
    RNS = np.array(RN)
    RU = np.array(RU)
    print(len(p))
    print(len(RNS))
    print(RU)

    while True:
        s = SVC()
        X = np.vstack((p, RNS))
        y = np.array([1] * len(p) + [-1] * len(RNS))
        s.fit(X, y)
        predicted = s.predict(RU)
        for e in predicted:
            if e == 1:

