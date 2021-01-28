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
    # if doPCA is True:                     # doPCA如果是TRUE，那么就会进行PCA降维
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
        col_label = 34  # 类标签位置
        texts = pd.read_csv(r'C:\Users\yyveggie\Desktop\UCI\dermatology.data', header=None)
        for indexs in texts.columns:
            if type(texts[indexs][0]) != np.int64:
                le = LabelEncoder()
                le.fit(texts[indexs])
                texts[indexs] = pd.Series(le.transform(texts[indexs]))  # 将字符串类型的属性值转换为数值类型
        labels = dict(Counter(texts[col_label]))  # dermatology数据集的列标签是第34列
        sorted_labels = sorted(labels.items(), key=lambda x: x[1], reverse=True)  # 按照样本数量排序，选择数量最多的两个类别
        labels_1 = sorted_labels[0][0]  # 样本数量最多的类别
        print("样本数量最多的类别（原类别）：", labels_1)
        labels_2 = sorted_labels[1][0]  # 样本数量第二多的类别
        print("样本数量第二多的类别（原类别）：", labels_2)

        texts.loc[texts[col_label] == labels_1, col_label] = 11
        texts.loc[texts[col_label] == labels_2, col_label] = 10
        texts.loc[texts[col_label] == 11, col_label] = 1  # 按照原文，将样本数量最多的标签改为1，即正例
        texts.loc[texts[col_label] == 10, col_label] = 0  # 按照原文，将样本数量第二多的标签改为0，即负例

        index_1 = [o for o, c in enumerate(texts[col_label]) if c == 1]  # 类别为1的样本集的索引
        texts_1 = texts.iloc[index_1]  # 类别为1的样本集
        print("最初有类别1样本数量为：", len(index_1))
        index_0 = [o for o, c in enumerate(texts[col_label]) if c == 0]  # 类别为0的样本集的索引
        texts_0 = texts.iloc[index_0]  # 类别为0的样本集
        print("最初有类别0样本数量为：", len(index_0))

        percent_p = 3  # p所占份数
        texts_1 = np.array_split(texts_1, 10)  # 将类别为1的样本集分成十份
        texts_0 = np.array_split(texts_0, 10)  # 将类别为0的样本集分成十份
        index_test = random.sample(range(len(texts_1)), 1)  # 随机选择1个测试集索引
        test = texts_1[index_test[0]].append(texts_0[index_test[0]])  # 测试集
        index_rest = sorted(set(range(len(texts_1))) - set(index_test))  # 除去测试集剩余索引
        index_p = random.sample(index_rest, percent_p)  # 随机选择3个p索引
        r = []
        p = []
        for i in index_p:
            r.append(texts_1[i])
        for i in range(len(r)):
            for j in range(len(r[i])):
                p.append(r[i].iloc[j].values)
        p = pd.DataFrame(p)  # p集合
        index_except_p = sorted(set(range(len(texts_1))) - set(index_test) - set(index_p))  # 除去测试集和p集合剩余索引
        r = []
        except_p = []
        for i in index_except_p:
            r.append(texts_1[i])
        for i in range(len(r)):
            for j in range(len(r[i])):
                except_p.append(r[i].iloc[j].values)  # 加入u的正例集合
        u = pd.DataFrame(except_p)
        r = []
        n = []
        for i in index_rest:
            r.append(texts_0[i])
        for i in range(len(r)):
            for j in range(len(r[i])):
                n.append(r[i].iloc[j].values)
        n = pd.DataFrame(n)  # 除去测试集后所有负例集合
        u = pd.concat([u, n])  # u集合

        u = u.drop([col_label], axis=1)  # 删除标签列
        p = p.drop([col_label], axis=1)  # 删除标签列
        y = pd.Series(test[col_label])  # 保存测试集标签
        test = test.drop([col_label], axis=1)  # 删除标签列
        data_P = p.values
        data_U = u.values
        test = test.values

        x = np.concatenate([data_P, data_U, test])
        t = np.vstack(pd.Series([1] * p.shape[0] + [0] * u.shape[0] + list(np.hstack(y))))
        doPCA = False

    else:
        raise ValueError

    return x, t, doPCA