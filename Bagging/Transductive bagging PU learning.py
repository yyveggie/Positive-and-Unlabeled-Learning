'''数据集是Libsvm格式'''

from __future__ import division, print_function
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from Conversion import Libsvm
import random
from svmutil import *

path = r'C:\Users\yyveggie\Desktop\UCI\Conversion\mushroom.libsvm'
f1 = []
accuracy = []

texts_1, texts_0 = Libsvm(path)

def Create_data(k, texts_1, texts_0):
    percent_p = 3                                                           # p所占份数
    texts_1 = np.array_split(texts_1, 10)                                   # 将类别为1的样本集分成十份
    texts_0 = np.array_split(texts_0, 10)                                   # 将类别为0的样本集分成十份
    test_x = list(texts_1[k]) + list(texts_0[k])                            # 测试集x，每一份每轮选一次
    test_y = list(len(texts_1[k]) * [1]) + list(len(texts_0[k]) * [0])      # 测试集y，正例为1，负例为0

    index_rest = sorted(set(range(10)) - set([k]))                          # 除去测试集剩余索引
    index_p = random.sample(index_rest, percent_p)                          # 随机选择percent_p个p索引
    texts_1 = np.array(texts_1)                                             # 转换为array格式可进行多维索引
    p = texts_1[index_p]                                                    # p集合

    index_except_p = sorted(set(range(10)) - set([k]) - set(index_p))       # 除去测试集和p集合剩余索引
    texts_0 = np.array(texts_0)                                             # 转换为array格式可进行多维索引
    u = list(texts_1[index_except_p]) + list(texts_0[index_rest])           # u集合
    return p, u, test_x, test_y

for k in range(10):                                        # 10折交叉
    p, u, test_x, test_y = Create_data(k, texts_1, texts_0)
    NP = sum([len(i) for i in p])                          # p的数量
    NU = sum([len(i) for i in u])                          # u的数量
    shape = len(test_y)                                    # test的数量

    T = 85                                      # 迭代次数
    K = NP                                      # 每次从U中bootstrap的样本数
    train_label = np.zeros(shape=(NP + K, ))    # 训练集标签初始化全为0（训练集=p+u）
    train_label[:NP] = 1.0                      # 前0~NP个P样本标签为1

    n_oob = np.zeros(shape=(shape, ))
    f_oob = np.zeros(shape=(shape, 2))

    p = np.array([j for i in p for j in i])
    u = np.array([j for i in u for j in i])
    test_x = np.array(test_x)
    test_y = np.array(test_y)

    for i in range(T):
        print("Iter：", i)
        # u中Bootstrap重采样样本的索引
        bootstrap_sample = np.random.choice(np.arange(NU), replace=True, size=K)
        # 训练样本集 = p + 从u中重采样的样本
        data_bootstrap = np.concatenate((p, u[bootstrap_sample]), axis=0)
        # Train model
        model = svm_train(np.array([1] * len(p) + [0] * len(bootstrap_sample)), data_bootstrap, '-b 1')
        # 预测测试集，并叠加每次预测的分数
        idx_alltest = sorted(set(range(len(test_y))))
        p_labels, p_acc, p_vals = svm_predict(test_y[idx_alltest], test_x[idx_alltest], model, '-b 1')
        f_oob[idx_alltest] += np.array(p_vals)[idx_alltest]
        n_oob[idx_alltest] += 1
    predict_proba = f_oob[:, 0] / n_oob         # 将第二列（属于P的概率）取平均

    y_pred = np.zeros(len(test_y))
    y_true = test_y
    for i in range(len(predict_proba)):
        if predict_proba[i] >= 0.5:
            y_pred[i] = 1

    f1.append(f1_score(y_true, y_pred, average='binary'))
    accuracy.append(accuracy_score(y_true, y_pred))

print("F1-Score：", np.mean(np.array(f1)))
print("Accuracy：", np.mean(np.array(accuracy)))