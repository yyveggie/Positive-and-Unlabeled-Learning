import numpy as np
import pandas as pd
from Conversion import CSV
import random
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

path = r'C:\Users\yyveggie\Desktop\UCI\Conversion\mushroom.csv'
percent_p = 3  # p所占份数
l = 0.05
alpha = 16
beta = 4
texts_1, texts_0 = CSV(path)
f1 = []
accuracy = []

def SplitData(k, texts_1, texts_0):
    print("Iter：", k)
    texts_1 = np.array_split(texts_1, 10)  # 将类别为1的样本集分成十份
    texts_0 = np.array_split(texts_0, 10)  # 将类别为0的样本集分成十份
    test_x = list(texts_1[k]) + list(texts_0[k])  # 测试集x，每一份每轮选一次
    test_y = list(len(texts_1[k]) * [1]) + list(len(texts_0[k]) * [0])  # 测试集y，正例为1，负例为0

    index_rest = sorted(set(range(10)) - {k})  # 除去测试集剩余索引
    index_p = random.sample(index_rest, percent_p)  # 随机选择percent_p个p索引
    texts_1 = np.array(texts_1)  # 转换为array格式可进行多维索引
    p = texts_1[index_p]  # p集合

    index_except_p = sorted(set(range(10)) - {k} - set(index_p))  # 除去测试集和p集合剩余索引
    texts_0 = np.array(texts_0)  # 转换为array格式可进行多维索引
    u = list(texts_1[index_except_p]) + list(texts_0[index_rest])  # u集合
    p = np.array([j for i in p for j in i])
    u = np.array([j for i in u for j in i])
    return p, u, test_x, test_y

def CentroidVector(samples):
    s = np.sum(o / np.linalg.norm(o) for o in samples) / len(samples)
    return s

def Rocchio(p, pn):
    p = np.array(p)
    pn = np.array(pn)
    CentroidP = CentroidVector(p)
    CentroidU = CentroidVector(pn)
    c_1 = CentroidP * alpha - CentroidU * beta
    c_0 = CentroidU * alpha - CentroidP * beta
    return c_1, c_0

def Similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * (np.linalg.norm(b)))

for k in range(10):
    COS_P, PN, Q = [], [], []
    p, u, test_x, test_y = SplitData(k, texts_1, texts_0)
    ### Find PN
    pr = CentroidVector(p)
    pr = pr / np.linalg.norm(pr)  # positive representative vector (pr)
    for j in p:  # 计算每个正例与pr的相似度
        COS_P.append(Similarity(j, pr))
    COS_P = sorted(COS_P, reverse=True)
    w = COS_P[round((1 - l) * len(p))]  # 相似度阈值
    for d in u:
        if Similarity(d, pr) < w:
            PN.append(d)
        else:
            Q.append(d)
    ### Find RN
    RN = []
    p = pd.DataFrame(p)
    PN = pd.DataFrame(PN)
    vec_p, vec_pn = Rocchio(p, PN)
    for g in u:
        if Similarity(g, vec_pn) > Similarity(g, vec_p):
            RN.append(g)
        else:
            Q.append(g)
    ###  Training
    Q = pd.DataFrame(Q)
    RN = pd.DataFrame(RN)

    svc_1 = SVC()
    svc_1.fit(pd.concat([p, RN]), pd.Series([1] * p.shape[0] + [0] * RN.shape[0]))

    lis = list(svc_1.predict(Q))
    W = pd.DataFrame([Q.iloc[o] for o in range(len(lis)) if lis[o] == 0])
    while W.shape[0] != 0:
        Q = pd.DataFrame([Q.iloc[o] for o in range(len(lis)) if lis[o] == 1])
        RN = pd.concat([RN, W])
        svc_last = SVC()
        svc_last.fit(pd.concat([p, RN]), pd.Series([1] * p.shape[0] + [0] * RN.shape[0]))
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
