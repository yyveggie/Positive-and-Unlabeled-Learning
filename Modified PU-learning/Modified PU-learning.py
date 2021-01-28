''' CSV文件格式 '''

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import f1_score
import pandas as pd
from Conversion import CSV
import random

f1_binary = []
accuracy = []
path = r'C:\Users\yyveggie\Desktop\UCI\Conversion\mushroom.csv'

texts_1, texts_0 = CSV(path)

def SplitData(k, texts_1, texts_0):
    percent_p = 3  # p所占份数
    texts_1 = np.array_split(texts_1, 10)  # 将类别为1的样本集分成十份
    texts_0 = np.array_split(texts_0, 10)  # 将类别为0的样本集分成十份
    test_x = list(texts_1[k]) + list(texts_0[k])  # 测试集x，每一份每轮选一次
    test_y = list(len(texts_1[k]) * [1]) + list(len(texts_0[k]) * [0])  # 测试集y，正例为1，负例为0

    index_rest = sorted(set(range(10)) - set([k]))  # 除去测试集剩余索引
    index_p = random.sample(index_rest, percent_p)  # 随机选择percent_p个p索引
    texts_1 = np.array(texts_1)  # 转换为array格式可进行多维索引
    p = texts_1[index_p]  # p集合

    index_except_p = sorted(set(range(10)) - set([k]) - set(index_p))  # 除去测试集和p集合剩余索引
    texts_0 = np.array(texts_0)  # 转换为array格式可进行多维索引
    u = list(texts_1[index_except_p]) + list(texts_0[index_rest])  # u集合
    p = np.array([j for i in p for j in i])
    u = np.array([j for i in u for j in i])
    return p, u, test_x, test_y

for k in range(10):
    p, u, test_x, test_y = SplitData(k, texts_1, texts_0)
    p = pd.DataFrame(p)
    u = pd.DataFrame(u)
    C = BernoulliNB()
    C.fit(pd.concat([p, u]), pd.Series([1] * p.shape[0] + [0] * u.shape[0]))
    lis = list(C.predict(u))
    Q = [u.iloc[i] for i in range(len(lis)) if lis[i] == 0]  # Q中保存被预测为负例的样本
    Q = pd.DataFrame(Q)
    RN = Q  # 将Q赋予为初始RN
    print("初始RN的数量为：", len(RN))
    Q_ = Q  # 初始设置两者相等，以防初始即跳出循环
    j = 1  # 计次用
    while len(Q) <= len(Q_) and p.shape[0] < len(RN):  # 如果当前被预测为负例的数量小于等于上一轮被预测为负例的数量，且P的数量少于当前RN数量，则进行以下循环
        print("iter:", k, j)
        Q_ = Q  # 当前Q被赋予为上一轮Q
        clf = BernoulliNB()
        clf.fit(pd.concat([p, RN]), pd.Series([1] * p.shape[0] + [0] * RN.shape[0]))  # 将p和当前RN训练一个分类器
        lis = list(clf.predict(RN))  # 分类器对当前RN进行预测
        Q = [RN.iloc[o] for o in range(len(lis)) if lis[o] == 0]  # 预测为负例的保留
        Q = pd.DataFrame(Q)  # 当前新Q
        RN = Q
        print("当前RN的数量为：", len(RN))
        j += 1
        if len(Q) == len(Q_):
            break
    y_pred = clf.predict(test_x)  # 停止条件终止之后，返回最后一个分类器，并对测试集分类
    f1_binary.append(f1_score(test_y, y_pred, average='binary'))
    accuracy.append(accuracy_score(test_y, y_pred))

print("F1-Score：", np.mean(np.array(f1_binary)))
print("Accuracy：", np.mean(np.array(accuracy)))