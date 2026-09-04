''' Helpers for the various data structures '''

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from collections import Counter
from svmutil import *

def Quantify(path):
    texts = pd.read_csv(path, header=None)
    for indexs in texts.columns:
        if type(texts[indexs][0]) != np.int64:
            le = LabelEncoder()
            le.fit(texts[indexs])
            texts[indexs] = pd.Series(le.transform(texts[indexs]))          # convert string-valued attributes to numeric
    return texts

def Libsvm(path):
    y, x = svm_read_problem(path)
    x = np.array(x)  # convert to numpy.array so multi-value indexing works
    labels = dict(Counter(y))
    sorted_labels = sorted(labels.items(), key=lambda x: x[1], reverse=True)  # sort by class size and keep the two largest classes
    labels_1 = sorted_labels[0][0]  # the largest class, taken as class 1
    print("largest class (original label):", labels_1, " count:", sorted_labels[0][1])
    labels_2 = sorted_labels[1][0]  # the second largest class, taken as class 0
    print("second largest class (original label):", labels_2, " count:", sorted_labels[1][1])

    index_1 = [o for o in range(len(y)) if y[o] == labels_1]  # indices of the class-1 samples
    texts_1 = x[index_1]  # the class-1 samples
    index_0 = [o for o in range(len(y)) if y[o] == labels_2]  # indices of the class-0 samples
    texts_0 = x[index_0]  # the class-0 samples
    return texts_1, texts_0

def SVMlight(path):
    y, x = svm_read_problem(path)
    x = np.array(x)  # convert to numpy.array so multi-value indexing works
    labels = dict(Counter(y))
    sorted_labels = sorted(labels.items(), key=lambda x: x[1], reverse=True)  # sort by class size and keep the two largest classes
    labels_1 = sorted_labels[0][0]  # the largest class, taken as class 1
    print("largest class (original label):", labels_1, " count:", sorted_labels[0][1])
    labels_2 = sorted_labels[1][0]  # the second largest class, taken as class 0
    print("second largest class (original label):", labels_2, " count:", sorted_labels[1][1])

    index_1 = [o for o in range(len(y)) if y[o] == labels_1]  # indices of the class-1 samples
    texts_1 = x[index_1]  # the class-1 samples
    index_0 = [o for o in range(len(y)) if y[o] == labels_2]  # indices of the class-0 samples
    texts_0 = x[index_0]  # the class-0 samples
    return texts_1, texts_0


def CSV(path):
    texts = pd.read_csv(path, header=None)  # the feature index is not read
    col_label = texts.shape[1] - 1  # label column
    labels = dict(Counter(texts[col_label]))
    sorted_labels = sorted(labels.items(), key=lambda x: x[1], reverse=True)  # sort by class size and keep the two largest classes
    labels_1 = sorted_labels[0][0]  # the largest class, taken as class 1
    print("largest class (original label):", labels_1, " count:", sorted_labels[0][1])
    labels_2 = sorted_labels[1][0]  # the second largest class, taken as class 0
    print("second largest class (original label):", labels_2, " count:", sorted_labels[1][1])

    index_1 = [o for o, c in enumerate(texts[col_label]) if c == labels_1]  # indices of the class-1 samples
    texts_1 = texts.iloc[index_1]  # the class-1 samples
    index_0 = [o for o, c in enumerate(texts[col_label]) if c == labels_2]  # indices of the class-0 samples
    texts_0 = texts.iloc[index_0]  # the class-0 samples
    texts_1 = texts_1.drop([col_label], axis=1)  # drop the column labels
    texts_0 = texts_0.drop([col_label], axis=1)  # drop the column labels
    texts_1 = np.array(texts_1)
    texts_0 = np.array(texts_0)
    return texts_1, texts_0