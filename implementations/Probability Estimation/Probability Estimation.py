''' CSV file format '''
import random
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import time
from Conversion import CSV

''' Create data '''

e1_f1_binary = []
e1_accuracy = []
e2_f1_binary = []
e2_accuracy = []
e3_f1_binary = []
e3_accuracy = []
path = r'data/mushroom.csv'

texts_1, texts_0 = CSV(path)

def SplitDate(k, texts_1, texts_0):
    percent_p = 3  # number of folds assigned to P
    texts_1 = np.array_split(texts_1, 10)  # split the class-1 samples into ten folds
    texts_0 = np.array_split(texts_0, 10)  # split the class-0 samples into ten folds
    test = pd.DataFrame(list(texts_1[k]) + list(texts_0[k]))  # test x; one fold is taken each round
    y = pd.Series(list(len(texts_1[k]) * [1]) + list(len(texts_0[k]) * [0]))  # test y; positive is 1, negative is 0

    index_rest = sorted(set(range(10)) - {k})  # indices left after removing the test fold
    index_p = random.sample(index_rest, percent_p)  # randomly choose percent_p indices for P
    texts_1 = np.array(texts_1)  # convert to an array so multi-dimensional indexing works
    p = texts_1[index_p]  # the P set

    index_except_p = sorted(set(range(10)) - {k} - set(index_p))  # indices left after removing the test fold and P
    texts_0 = np.array(texts_0)  # convert to an array so multi-dimensional indexing works
    u = list(texts_1[index_except_p]) + list(texts_0[index_rest])  # the U set
    p = pd.DataFrame(np.array([j for i in p for j in i]))
    u = pd.DataFrame(np.array([j for i in u for j in i]))
    p['y'] = 1
    u['y'] = pd.Series([1] * len([j for i in list(texts_1[index_except_p]) for j in i]) + [0] * len([j for i in list(texts_0[index_rest]) for j in i]))

    test['y'] = y
    return u, p, test

for k in range(10):
    u, p, test = SplitDate(k, texts_1, texts_0)

    # an ordinary classifier on ordinary data, as an example
    classifier = SVC

    # generate some additional undefined data
    p["s"] = p["y"] * np.random.randint(1, 2, len(p))      # only positives are labeled, so s=1 implies y=1, while s=0 leaves y at 0 or 1
    u["s"] = u["y"] * np.random.randint(0, 1, len(u))

    # split the data
    y_train = p[['y', 's']].append(u[['y', 's']])                                   # take out the y labels first
    X_train = p.drop(['y', 's'], axis=1).append(u.drop(['y', 's'], axis=1))         # assemble the training set
    y_test = test
    X_test = test.drop(['y'], axis=1)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.33, random_state=42)    # split the training set into train and validation

    # fit g, which returns the probability that a sample is labeled, g(x) = p(s = 1|x); g is the non-traditional classifier
    CL_g = classifier(probability=True, gamma='auto')
    CL_g.fit(X_train, y_train["s"])                         # train the classifier using the s column as the label

    # Estimating c
    X_val_p = X_val.loc[X_val.index[y_val["s"] == 1]]       # s=1, the validation samples carrying s=1
    d_p = CL_g.predict_proba(X_val_p)[:, 1]                 # score X_val_p
    e1 = d_p.sum()/len(X_val_p)
    print("e1：", e1)

    d_v = CL_g.predict_proba(X_val)[:, 1]                   # score the whole validation set
    e2 = d_p.sum()/d_v.sum()
    print("e2：", e2)

    e3 = d_v.max()
    print("e3：", e3)

    # First way to create f classificator
    CL_f = classifier(probability=True, gamma='auto')
    CL_f.fit(X_train, y_train["s"])                 # train g using the s column as the label
    c = e1                                          # e2 or e3; e1 is used because the paper reports it works best
    print("current c:", c)
    y_pred = np.array(CL_f.predict_proba(X_test)[:, 1]/c > 0.5, dtype=int)      # prediction of g divided by c gives the prediction of f
    print("accuracy of f at the current c:", accuracy_score(y_test["y"], y_pred))

    # Second way to improve classifier f

    # The way, that was used by the authors of article is to create weights
    # of each sample build on g prediction for value and parameter c
    # following the paper, g(x) supplies a weight for every sample,
    # after which a weighted classifier does better
    def create_new_train(e, X_train, y_train, CL):
        X_train_new = pd.DataFrame(X_train.loc[y_train.index[y_train["s"] == 1]])   # labeled samples
        X_train_new["w"] = 1    # labeled samples carry weight 1
        y_train_new = pd.DataFrame([1] * len(X_train_new), columns=["s"])       # the s label of the labeled samples
        X_train_1 = pd.DataFrame(X_train.loc[y_train.index[y_train["s"] == 0]])     # unlabeled samples
        g_x = CL.predict_proba(X_train_1)[:, 1]
        X_train_1["w"] = (1 - e) / e * g_x / (1 - g_x)
        y_train_1 = pd.DataFrame([1] * len(X_train_1), columns=["s"])
        X_train_2 = pd.DataFrame(X_train.loc[y_train.index[y_train["s"] == 0]])
        g_x = CL.predict_proba(X_train_2)[:, 1]
        X_train_2["w"] = 1 - (1 - e) / e * g_x / (1 - g_x)
        y_train_2 = pd.DataFrame([0] * len(X_train_2), columns=["s"])
        X_train_new = X_train_new.append(X_train_1, ignore_index=True, sort=False)
        X_train_new = X_train_new.append(X_train_2, ignore_index=True, sort=False)
        y_train_new = y_train_new.append(y_train_1, ignore_index=True, sort=False)
        y_train_new = y_train_new.append(y_train_2, ignore_index=True, sort=False)
        return X_train_new, y_train_new

    # e1
    CL_f = classifier(probability=True, gamma='auto')
    X_train_new, y_train_new = create_new_train(e1, X_train, y_train, CL_g)
    X_new = X_train_new.drop(['w'], axis=1)
    CL_f.fit(X_new, y_train_new["s"], sample_weight=X_train_new['w'])
    y_true = y_test["y"]
    y_pred = CL_f.predict(X_test)

    e1_f1_binary.append(f1_score(y_true, y_pred, average='binary'))
    e1_accuracy.append(accuracy_score(y_true, y_pred))

    # e2
    CL_f = classifier(probability=True, gamma='auto')
    X_train_new, y_train_new = create_new_train(e2, X_train, y_train, CL_g)
    X_new = X_train_new.drop(['w'], axis=1)
    CL_f.fit(X_new, y_train_new["s"], sample_weight=X_train_new['w'])
    y_true = y_test["y"]
    y_pred = CL_f.predict(X_test)

    e2_f1_binary.append(f1_score(y_true, y_pred, average='binary'))
    e2_accuracy.append(accuracy_score(y_true, y_pred))

    # e3
    CL_f = classifier(probability=True, gamma='auto')
    X_train_new, y_train_new = create_new_train(e3, X_train, y_train, CL_g)
    X_new = X_train_new.drop(['w'], axis=1)
    CL_f.fit(X_new, y_train_new["s"], sample_weight=X_train_new['w'])
    y_true = y_test["y"]
    y_pred = CL_f.predict(X_test)

    e3_f1_binary.append(f1_score(y_true, y_pred, average='binary'))
    e3_accuracy.append(accuracy_score(y_true, y_pred))

print("F1-Score for e1：", np.mean(np.array(e1_f1_binary)))
print("Accuracy for e1：", np.mean(np.array(e1_accuracy)))
print("*" * 50)
print("F1-Score for e2：", np.mean(np.array(e2_f1_binary)))
print("Accuracy for e2：", np.mean(np.array(e2_accuracy)))
print("*" * 50)
print("F1-Score for e3：", np.mean(np.array(e3_f1_binary)))
print("Accuracy for e3：", np.mean(np.array(e3_accuracy)))