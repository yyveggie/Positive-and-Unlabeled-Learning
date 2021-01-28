''' 用来处理各种数据结构 '''

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
            texts[indexs] = pd.Series(le.transform(texts[indexs]))          # 将字符串类型的属性值转换为数值类型
    return texts

def Libsvm(path):
    y, x = svm_read_problem(path)
    x = np.array(x)  # 转换为numpy.array模型可以进行多值索引
    labels = dict(Counter(y))
    sorted_labels = sorted(labels.items(), key=lambda x: x[1], reverse=True)  # 按照样本数量排序，选择数量最多的两个类别
    labels_1 = sorted_labels[0][0]  # 样本数量最多的类别（记为类别1）
    print("样本数量最多的类别（原类别）：", labels_1, "其数量为：", sorted_labels[0][1])
    labels_2 = sorted_labels[1][0]  # 样本数量第二多的类别（记为类别0）
    print("样本数量第二多的类别（原类别）：", labels_2, "其数量为：", sorted_labels[1][1])

    index_1 = [o for o in range(len(y)) if y[o] == labels_1]  # 类别为1的样本集的索引
    texts_1 = x[index_1]  # 类别为1的样本集
    index_0 = [o for o in range(len(y)) if y[o] == labels_2]  # 类别为0的样本集的索引
    texts_0 = x[index_0]  # 类别为0的样本集
    return texts_1, texts_0

def SVMlight(path):
    y, x = svm_read_problem(path)
    x = np.array(x)  # 转换为numpy.array模型可以进行多值索引
    labels = dict(Counter(y))
    sorted_labels = sorted(labels.items(), key=lambda x: x[1], reverse=True)  # 按照样本数量排序，选择数量最多的两个类别
    labels_1 = sorted_labels[0][0]  # 样本数量最多的类别（记为类别1）
    print("样本数量最多的类别（原类别）：", labels_1, "其数量为：", sorted_labels[0][1])
    labels_2 = sorted_labels[1][0]  # 样本数量第二多的类别（记为类别0）
    print("样本数量第二多的类别（原类别）：", labels_2, "其数量为：", sorted_labels[1][1])

    index_1 = [o for o in range(len(y)) if y[o] == labels_1]  # 类别为1的样本集的索引
    texts_1 = x[index_1]  # 类别为1的样本集
    index_0 = [o for o in range(len(y)) if y[o] == labels_2]  # 类别为0的样本集的索引
    texts_0 = x[index_0]  # 类别为0的样本集
    return texts_1, texts_0


def CSV(path):
    texts = pd.read_csv(path, header=None)  # 特征索引不读取
    col_label = texts.shape[1] - 1  # 标签列
    labels = dict(Counter(texts[col_label]))
    sorted_labels = sorted(labels.items(), key=lambda x: x[1], reverse=True)  # 按照样本数量排序，选择数量最多的两个类别
    labels_1 = sorted_labels[0][0]  # 样本数量最多的类别（记为类别1）
    print("样本数量最多的类别（原类别）：", labels_1, "其数量为：", sorted_labels[0][1])
    labels_2 = sorted_labels[1][0]  # 样本数量第二多的类别（记为类别0）
    print("样本数量第二多的类别（原类别）：", labels_2, "其数量为：", sorted_labels[1][1])

    index_1 = [o for o, c in enumerate(texts[col_label]) if c == labels_1]  # 类别为1的样本集的索引
    texts_1 = texts.iloc[index_1]  # 类别为1的样本集
    index_0 = [o for o, c in enumerate(texts[col_label]) if c == labels_2]  # 类别为0的样本集的索引
    texts_0 = texts.iloc[index_0]  # 类别为0的样本集
    texts_1 = texts_1.drop([col_label], axis=1)  # 删除列标签
    texts_0 = texts_0.drop([col_label], axis=1)  # 删除列标签
    texts_1 = np.array(texts_1)
    texts_0 = np.array(texts_0)
    return texts_1, texts_0