'''CSV文件格式'''

from Conversion import CSV
import numpy as np
import random
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import pandas as pd

f1_binary = []
accuracy = []
T = 4.8  # 阈值
k = 5

path = r'C:\Users\yyveggie\Desktop\UCI\Conversion\mushroom.csv'

texts_1, texts_0 = CSV(path)

def SplitData(t, texts_1, texts_0):  # 10折交叉
    percent_p = 3  # p所占份数
    texts_1 = np.array_split(texts_1, 10)  # 将类别为1的样本集分成十份
    texts_0 = np.array_split(texts_0, 10)  # 将类别为0的样本集分成十份
    test_x = list(texts_1[t]) + list(texts_0[t])  # 测试集x，每一份每轮选一次
    test_y = list(len(texts_1[t]) * [1]) + list(len(texts_0[t]) * [0])  # 测试集y，正例为1，负例为0

    index_rest = sorted(set(range(10)) - set([t]))  # 除去测试集剩余索引
    index_p = random.sample(index_rest, percent_p)  # 随机选择percent_p个p索引
    texts_1 = np.array(texts_1)                                                # 转换为array格式可进行多维索引
    p = texts_1[index_p]                                                       # p集合

    index_except_p = sorted(set(range(10)) - set([t]) - set(index_p))          # 除去测试集和p集合剩余索引
    texts_0 = np.array(texts_0)                                                # 转换为array格式可进行多维索引
    u = list(texts_1[index_except_p]) + list(texts_0[index_rest])              # u集合
    p = np.array([j for i in p for j in i])
    u = np.array([j for i in u for j in i])
    return p, u, test_x, test_y

for o in range(10):
    print("Iter：", o, '\n', '*' * 50)
    RN = []
    R_ = []
    p, u, test_x, test_y = SplitData(o, texts_1, texts_0)
    for i in range(len(u)):
        sim = {}
        sum_sim = []
        ui = u[i]
        for j in range(len(p)):
            pj = p[j]
            cos = np.dot(ui, pj) / (np.linalg.norm(ui) * (np.linalg.norm(pj)))  # 计算余弦距离
            sim[j] = cos
        sim = sorted(sim.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)  # 将字典按照余弦值排序
        for n in range(k):  # 选择前k个最大的索引
            sum_sim.append(sim[n][1])
        value = sum(sum_sim)   # 前k个相似度相加
        wi = value - T

        if wi < 0:
            RN.append(u[i])
        else:
            R_.append(u[i])
    p = pd.DataFrame(p)
    u = pd.DataFrame(u)
    test_x = pd.DataFrame(test_x)
    test_y = pd.DataFrame(test_y)
    RN = pd.DataFrame(RN)
    R_ = pd.DataFrame(R_)

    x_train = pd.concat([p, RN])
    y_train = pd.Series(list([1] * len(p)) + list([0] * len(RN)))

    svc = SVC(kernel='linear')
    svc.fit(x_train, y_train)
    lis = list(svc.predict(R_))
    W = pd.DataFrame([R_.iloc[o] for o in range(len(lis)) if lis[o] == 0])  # 预测为负例的保留
    Q = pd.DataFrame([R_.iloc[o] for o in range(len(lis)) if lis[o] == 1])
    while len(W) > 0:
        print("len(W):", len(W) )
        R = Q
        RN = pd.concat([RN, W])
        clf = SVC(kernel='linear')
        X = np.vstack((p, RN))
        y = pd.Series(list([1] * len(p)) + list([0] * len(RN)))
        clf.fit(X, y)
        llis = list(clf.predict(R))
        W = pd.DataFrame([R.iloc[o] for o in range(len(llis)) if llis[o] == 0])
        Q = pd.DataFrame([R.iloc[o] for o in range(len(llis)) if llis[o] == 1])

    y_pred = pd.DataFrame(clf.predict(test_x))  # 停止条件终止之后，返回最后一个分类器，并对测试集分类
    f1_binary.append(f1_score(test_y, y_pred, average='binary'))
    accuracy.append(accuracy_score(test_y, y_pred))
    print(f1_score(test_y, y_pred, average='binary'))
    print(accuracy_score(test_y, y_pred))

print("F1-Score：", np.mean(np.array(f1_binary)))
print("Accuracy：", np.mean(np.array(accuracy)))