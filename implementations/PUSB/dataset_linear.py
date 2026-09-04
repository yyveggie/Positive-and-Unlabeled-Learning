import numpy as np

import chainer
from sklearn.decomposition import PCA
from sklearn.datasets import load_svmlight_file
import numpy as np
import pandas as pd
from collections import Counter
import random
from sklearn.preprocessing import LabelEncoder


def make_data(datatype='uci', seed=2018, pca_dim=100):
    print("data_name", datatype)
    x, t, doPCA = get_data(datatype)
    print("x_shape", x.shape)
    print("t_shape", t.shape)
    # if doPCA is True:                     # when doPCA is True, reduce the dimension by PCA
    # pca = PCA(n_components=pca_dim)
    # pca.fit(x.T)
    # x = pca.components_.T
    return x, t


def get_data(datatype):
    doPCA = False
    if datatype == "mushroom":
        x, t = load_svmlight_file("dataset/mushrooms.txt")
        x = x.toarray()
        t[t == 1] = 0
        t[t == 2] = 1
        doPCA = True

    elif datatype == "waveform":
        data = np.loadtxt('dataset/waveform.txt', delimiter=',')
        x, t = data[:, :-1], data[:, -1]
        t[t == 2] = 0

    elif datatype == "shuttle":
        x_train, t_train = load_svmlight_file('dataset/shuttle.scale.txt')
        x_train = x_train.toarray()
        x_test, t_test = load_svmlight_file('dataset/shuttle.scale.t.txt')
        x_test = x_test.toarray()
        x = np.concatenate([x_train, x_test])
        t = np.concatenate([t_train, t_test])
        t[~(t == 1)] = 0

    elif datatype == "pageblocks":
        data = np.loadtxt('dataset/page-blocks.txt')
        x, t = data[:, :-1], data[:, -1]
        t[~(t == 1)] = 0

    elif datatype == "digits":
        train, test = chainer.datasets.get_mnist()
        x_train, t_train = train.datasets
        x_test, t_test = test.datasets
        x = np.concatenate([x_train, x_test])
        t = np.concatenate([t_train, t_test])
        t[t % 2 == 0] = 0
        t[t % 2 == 1] = 1
        doPCA = True

    elif datatype == "spambase":
        data = np.loadtxt('dataset/spambase.data.txt', delimiter=',')
        x, t = data[:, :-1], data[:, -1]

    elif datatype == "usps":
        x_train, t_train = load_svmlight_file('dataset/usps')
        x_train = x_train.toarray()
        x_test, t_test = load_svmlight_file('dataset/usps.t')
        x_test = x_test.toarray()
        x = np.concatenate([x_train, x_test])
        t = np.concatenate([t_train, t_test])
        t[t % 2 == 0] = 0
        t[t % 2 == 1] = 1
        print(np.mean(t))
        doPCA = True

    elif datatype == "connect-4":
        x, t = load_svmlight_file('dataset/connect-4.txt')
        x = x.toarray()
        t[t == -1] = 0
        print(np.mean(t))
        doPCA = True

    elif datatype == "protein":
        x_train, t_train = load_svmlight_file('dataset/protein.txt')
        x_train = x_train.toarray()
        x_test, t_test = load_svmlight_file('dataset/protein.t.txt')
        x_test = x_test.toarray()
        x = np.concatenate([x_train, x_test])
        t = np.concatenate([t_train, t_test])
        t[~(t == 1)] = 0
        print(np.mean(t))
        doPCA = True

    elif datatype == "ijcnn1":
        x, t = load_svmlight_file('./dataset/ijcnn1')
        x = x.toarray()
        t[~(t == 1)] = 0
        print(np.mean(t))
        doPCA = False

    elif datatype == "w1a":
        x_train, t_train = load_svmlight_file('./dataset/w1a')
        x_train = x_train.toarray()
        x_test, t_test = load_svmlight_file('./dataset/w1a.t')
        x_test = x_test.toarray()
        x = np.concatenate([x_train, x_test])
        t = np.concatenate([t_train, t_test])
        t[~(t == 1)] = 0
        print(np.mean(t))
        doPCA = False

    elif datatype == 'uci':
        col_label = 34  # position of the class label
        texts = pd.read_csv(r'data/dermatology.data', header=None)
        for indexs in texts.columns:
            if type(texts[indexs][0]) != np.int64:
                le = LabelEncoder()
                le.fit(texts[indexs])
                texts[indexs] = pd.Series(le.transform(texts[indexs]))  # convert string-valued attributes to numeric
        labels = dict(Counter(texts[col_label]))  # the dermatology label sits in column 34
        sorted_labels = sorted(labels.items(), key=lambda x: x[1], reverse=True)  # sort by class size and keep the two largest classes
        labels_1 = sorted_labels[0][0]  # the largest class
        print("largest class (original label):", labels_1)
        labels_2 = sorted_labels[1][0]  # the second largest class
        print("second largest class (original label):", labels_2)

        texts.loc[texts[col_label] == labels_1, col_label] = 11
        texts.loc[texts[col_label] == labels_2, col_label] = 10
        texts.loc[texts[col_label] == 11, col_label] = 1  # as in the paper, relabel the largest class 1, the positive class
        texts.loc[texts[col_label] == 10, col_label] = 0  # as in the paper, relabel the second largest class 0, the negative class

        index_1 = [o for o, c in enumerate(texts[col_label]) if c == 1]  # indices of the class-1 samples
        texts_1 = texts.iloc[index_1]  # the class-1 samples
        print("initial number of class-1 samples:", len(index_1))
        index_0 = [o for o, c in enumerate(texts[col_label]) if c == 0]  # indices of the class-0 samples
        texts_0 = texts.iloc[index_0]  # the class-0 samples
        print("initial number of class-0 samples:", len(index_0))

        percent_p = 3  # number of folds assigned to P
        texts_1 = np.array_split(texts_1, 10)  # split the class-1 samples into ten folds
        texts_0 = np.array_split(texts_0, 10)  # split the class-0 samples into ten folds
        index_test = random.sample(range(len(texts_1)), 1)  # randomly choose one test index
        test = texts_1[index_test[0]].append(texts_0[index_test[0]])  # the test set
        index_rest = sorted(set(range(len(texts_1))) - set(index_test))  # indices left after removing the test fold
        index_p = random.sample(index_rest, percent_p)  # randomly choose three P indices
        r = []
        p = []
        for i in index_p:
            r.append(texts_1[i])
        for i in range(len(r)):
            for j in range(len(r[i])):
                p.append(r[i].iloc[j].values)
        p = pd.DataFrame(p)  # the P set
        index_except_p = sorted(set(range(len(texts_1))) - set(index_test) - set(index_p))  # indices left after removing the test fold and P
        r = []
        except_p = []
        for i in index_except_p:
            r.append(texts_1[i])
        for i in range(len(r)):
            for j in range(len(r[i])):
                except_p.append(r[i].iloc[j].values)  # the positives folded into U
        u = pd.DataFrame(except_p)
        r = []
        n = []
        for i in index_rest:
            r.append(texts_0[i])
        for i in range(len(r)):
            for j in range(len(r[i])):
                n.append(r[i].iloc[j].values)
        n = pd.DataFrame(n)  # all negatives left after removing the test set
        u = pd.concat([u, n])  # the U set

        u = u.drop([col_label], axis=1)  # drop the label column
        p = p.drop([col_label], axis=1)  # drop the label column
        y = pd.Series(test[col_label])  # save the test labels
        test = test.drop([col_label], axis=1)  # drop the label column
        data_P = p.values
        data_U = u.values
        test = test.values

        x = np.concatenate([data_P, data_U, test])
        t = np.vstack(pd.Series([1] * p.shape[0] + [0] * u.shape[0] + list(np.hstack(y))))
        doPCA = False

    else:
        raise ValueError

    return x, t, doPCA