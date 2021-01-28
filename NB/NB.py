'''CSV格式'''

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import f1_score
import random
from Conversion import CSV

f1 = []
accuracy = []
path = r'C:\Users\yyveggie\Desktop\UCI\Conversion\mushroom.csv'

texts_1, texts_0 = CSV(path)

def SplitData(k, texts_1, texts_0):                                            # 10折交叉
    percent_p = 3                                                              # p所占份数
    texts_1 = np.array_split(texts_1, 10)                                      # 将类别为1的样本集分成十份
    texts_0 = np.array_split(texts_0, 10)                                      # 将类别为0的样本集分成十份
    test_x = list(texts_1[k]) + list(texts_0[k])                               # 测试集x，每一份每轮选一次
    test_y = list(len(texts_1[k]) * [1]) + list(len(texts_0[k]) * [0])         # 测试集y，正例为1，负例为0

    index_rest = sorted(set(range(10)) - {k})                             # 除去测试集剩余索引
    index_p = random.sample(index_rest, percent_p)                             # 随机选择percent_p个p索引
    texts_1 = np.array(texts_1)                                                # 转换为array格式可进行多维索引
    p = texts_1[index_p]                                                       # p集合

    index_except_p = sorted(set(range(10)) - {k} - set(index_p))          # 除去测试集和p集合剩余索引
    texts_0 = np.array(texts_0)                                                # 转换为array格式可进行多维索引
    u = list(texts_1[index_except_p]) + list(texts_0[index_rest])              # u集合
    p = np.array([j for i in p for j in i])
    u = np.array([j for i in u for j in i])
    return p, u, test_x, test_y

for k in range(10):
    print("Iter：", k)
    p, u, test_x, test_y = SplitData(k, texts_1, texts_0)
    C = GaussianNB()
    C.fit(np.vstack((p, u)), [1] * p.shape[0] + [0] * u.shape[0])
    y_pred = C.predict(test_x)

    f1.append(f1_score(test_y, y_pred, average='binary'))
    accuracy.append(accuracy_score(test_y, y_pred))

print("F1-Score：", np.mean(np.array(f1)))
print("Accuracy：", np.mean(np.array(accuracy)))