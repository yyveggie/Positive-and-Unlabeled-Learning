'''CSV文件格式'''

from Conversion import CSV
from sklearn.svm import SVC
import numpy as np
import random

f1 = []
accuracy = []
path = r'C:\Users\yyveggie\Desktop\UCI\Conversion\mushroom.csv'

k = 5   # k近邻
T = 4.8  # 阈值

texts_1, texts_0 = CSV(path)

def SplitData(k, texts_1, texts_0):                                            # 10折交叉
    percent_p = 3                                                              # p所占份数
    texts_1 = np.array_split(texts_1, 10)                                      # 将类别为1的样本集分成十份
    texts_0 = np.array_split(texts_0, 10)                                      # 将类别为0的样本集分成十份
    test_x = list(texts_1[k]) + list(texts_0[k])                               # 测试集x，每一份每轮选一次
    test_y = list(len(texts_1[k]) * [1]) + list(len(texts_0[k]) * [0])         # 测试集y，正例为1，负例为0

    index_rest = sorted(set(range(10)) - set([k]))                             # 除去测试集剩余索引
    index_p = random.sample(index_rest, percent_p)                             # 随机选择percent_p个p索引
    texts_1 = np.array(texts_1)                                                # 转换为array格式可进行多维索引
    p = texts_1[index_p]                                                       # p集合

    index_except_p = sorted(set(range(10)) - set([k]) - set(index_p))          # 除去测试集和p集合剩余索引
    texts_0 = np.array(texts_0)                                                # 转换为array格式可进行多维索引
    u = list(texts_1[index_except_p]) + list(texts_0[index_rest])              # u集合
    p = np.array([j for i in p for j in i])
    u = np.array([j for i in u for j in i])
    return p, u, test_x, test_y

for k in range(10):
    print("Iter：", k, '\n', '*' * 50)
    p, u, test_x, test_y = SplitData(k, texts_1, texts_0)
    RN = []
    RU = []
    V = []
    for i in range(len(u)):
        sim = {}
        sum_sim = []
        ui = u[i]
        for j in range(len(p)):
            pj = p[j]
            cos = np.dot(ui, pj) / (np.linalg.norm(ui) * (np.linalg.norm(pj)))  # 计算余弦距离
            sim[i] = cos
        sim = sorted(sim.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)  # 将字典按照余弦值排序
        for n in range(k):  # 选择前k个最大的索引
            sum_sim.append(sim[n][1])
        value = sum(sum_sim)  # 前k个相似度相加
        wi = value - T

        if wi < 0:
            RN.append(u[i])
        else:
            RU.append(u[i])
    RNS = np.array(RN)
    RU = np.array(RU)
    print(len(p))
    print(len(RNS))
    print(RU)

    while True:
        s = SVC()
        X = np.vstack((p, RNS))
        y = np.array([1] * len(p) + [-1] * len(RNS))
        s.fit(X, y)
        predicted = s.predict(RU)
        for e in predicted:
            if e == 1:

