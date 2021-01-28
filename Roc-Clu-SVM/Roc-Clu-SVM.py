import numpy as np
import pandas as pd
from Conversion import CSV
import random
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.cluster import KMeans

path = r'C:\Users\yyveggie\Desktop\UCI\Conversion\mushroom.csv'
percent_p = 3  # p所占份数
alpha = 16
beta = 4
k = 14
texts_1, texts_0 = CSV(path)
f1, accuracy = [], []

def SplitData(q, texts_1, texts_0):
    print("Iter：", q)
    texts_1 = np.array_split(texts_1, 10)  # 将类别为1的样本集分成十份
    texts_0 = np.array_split(texts_0, 10)  # 将类别为0的样本集分成十份
    test_x = list(texts_1[q]) + list(texts_0[q])  # 测试集x，每一份每轮选一次
    test_y = list(len(texts_1[q]) * [1]) + list(len(texts_0[q]) * [0])  # 测试集y，正例为1，负例为0

    index_rest = sorted(set(range(10)) - {q})  # 除去测试集剩余索引
    index_p = random.sample(index_rest, percent_p)  # 随机选择percent_p个p索引
    texts_1 = np.array(texts_1)  # 转换为array格式可进行多维索引
    p = texts_1[index_p]  # p集合

    index_except_p = sorted(set(range(10)) - {q} - set(index_p))  # 除去测试集和p集合剩余索引
    texts_0 = np.array(texts_0)  # 转换为array格式可进行多维索引
    u = list(texts_1[index_except_p]) + list(texts_0[index_rest])  # u集合
    p = np.array([j for i in p for j in i])
    u = np.array([j for i in u for j in i])
    return p, u, test_x, test_y

def CentroidVector(samples):
    s = np.sum(o / np.linalg.norm(o) for o in samples) / len(samples)
    return s

def Rocchio(p, u):
    CentroidP = CentroidVector(p)
    CentroidU = CentroidVector(u)
    c_1 = CentroidP * alpha - CentroidU * beta
    c_0 = CentroidU * alpha - CentroidP * beta
    return c_1, c_0

for q in range(10):
    RN, RN_, Q = [], [], []
    p, u, test_x, test_y = SplitData(q, texts_1, texts_0)
    c_1, c_0 = Rocchio(p, u)
    for j in u:
        cos_1 = np.dot(c_1, j) / (np.linalg.norm(c_1) * (np.linalg.norm(j)))
        cos_0 = np.dot(c_0, j) / (np.linalg.norm(c_0) * (np.linalg.norm(j)))
        if cos_1 <= cos_0:
            RN.append(j)

    y_pred = pd.DataFrame(KMeans(n_clusters=k).fit_predict(RN), columns=["y"])
    RN = pd.DataFrame(RN)
    N, P = [], []
    for d in range(k):
        N.append(alpha * CentroidVector(np.array(RN[(y_pred.y == d)])) - beta * CentroidVector(p))
        P.append(alpha * CentroidVector(p) - beta * CentroidVector(np.array(RN[(y_pred.y == d)])))
    RN = np.array(RN)
    for a in RN:
        COS_P, COS_N = [], []
        for b in P:
            COS_P.append(np.dot(a, b) / (np.linalg.norm(a) * (np.linalg.norm(b))))
        for c in N:
            COS_N.append(np.dot(a, c) / (np.linalg.norm(a) * (np.linalg.norm(c))))
        if max(COS_P) <= max(COS_N):
            RN_.append(a)
        else:
            Q.append(a)

    p = pd.DataFrame(p)
    RN_ = pd.DataFrame(RN_)
    Q = pd.DataFrame(Q)

    svc_1 = SVC()
    svc_1.fit(pd.concat([p, RN_]), pd.Series([1] * p.shape[0] + [0] * RN_.shape[0]))

    lis = list(svc_1.predict(Q))
    W = pd.DataFrame([Q.iloc[o] for o in range(len(lis)) if lis[o] == 0])
    while W.shape[0] != 0:
        Q = pd.DataFrame([Q.iloc[o] for o in range(len(lis)) if lis[o] == 1])
        RN_ = pd.concat([RN_, W])
        svc_last = SVC()
        svc_last.fit(pd.concat([p, RN_]), pd.Series([1] * p.shape[0] + [0] * RN_.shape[0]))
        lis = list(svc_last.predict(Q))
        W = pd.DataFrame([Q.iloc[o] for o in range(len(lis)) if lis[o] == 0])

    lastpred = list(svc_last.predict(p))
    if sum(lastpred) <= len(lastpred) * 0.95:
        svc = svc_1
    else:
        svc = svc_last

    y_pred = svc.predict(test_x)

    f1.append(f1_score(test_y, y_pred, average='binary'))
    accuracy.append(accuracy_score(test_y, y_pred))

print("F1-Score：", np.mean(np.array(f1)))
print("Accuracy：", np.mean(np.array(accuracy)))