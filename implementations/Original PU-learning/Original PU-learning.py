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
s = time.time()
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

### Two-Step ###

for k in range(10):
    print("iter", k)
    p, u, test_x, test_y = SplitData(k, texts_1, texts_0)
    p = pd.DataFrame(p)
    u = pd.DataFrame(u)
    # first train a classifier with all of U as negative and P as positive
    X_train = pd.concat([p, u])
    y_train = pd.Series([1] * len(p) + [0] * len(u))
    rf = GaussianNB()
    rf.fit(X_train, y_train)

    # score U so that every sample there gets a refreshed label
    pred = rf.predict(X_train)

    # STEP 1

    # Classifier to be used for step 2
    rf2 = GaussianNB()

    num_p_2 = len(X_train)                # start the stopping value large so the first round cannot terminate

    for i in range(1000):
        num_p = list(pred).count(1)       # count identified in this round

        if num_p < num_p_2:
            num_p_2 = num_p                                                # the current count becomes the previous count
            stand = pd.Series([n == 0 for n in pred])                      # find the samples in U still judged negative (n == 0)
            U = pd.DataFrame(X_train.values[stand])                        # keep in U those still judged negative (n == 0)
            print("%d left" % len(U) + ' in U')
            print('Step 1 labeled %d new positives and %d new negatives.' % (len([k for k in stand if k == 1]), len([o for o in stand if o == 0])))
            print('Doing step 2 ... ')

            # STEP 2
            U_y = len(U) * [0]
            X = pd.DataFrame(p).append(pd.DataFrame(U))
            X_y = pd.Series(len(p) * [1] + U_y)

            rf2 = GaussianNB()                  # retrain on the original P and the new U
            rf2.fit(X, X_y)

            pred = rf2.predict(U)               # score the new U with the resulting classifier
            num_p = list(pred).count(1)         # count how many are classified as P
            X_train = U

        else:
            y_pred = rf2.predict(test_x)        # once the stopping rule fires, return the last classifier and score the test set
            f1_binary.append(f1_score(test_y, y_pred, average='binary'))
            accuracy.append(accuracy_score(test_y, y_pred))
            break

print("F1-Score：", np.mean(np.array(f1_binary)))
print("Accuracy：", np.mean(np.array(accuracy)))
print("Time：", time.time() - s)