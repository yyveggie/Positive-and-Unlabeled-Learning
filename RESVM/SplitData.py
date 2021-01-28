'''Libsvm数据格式'''

import numpy as np
import random

# 先用这个代码将原数据集划分为10折，并保存到emsemblesvm的目录，再调用Cygwin来运行emsemblesvm训练和预测样本

from Conversion import Libsvm

path = r'C:\Users\yyveggie\Desktop\UCI\Conversion\mushroom.libsvm'
percent_p = 3  # p所占份数
texts_1, texts_0 = Libsvm(path)
texts_1 = np.array_split(texts_1, 10)  # 将类别为1的样本集分成十份
texts_0 = np.array_split(texts_0, 10)  # 将类别为0的样本集分成十份

for k in range(10):
    test = list(texts_1[k]) + list(texts_0[k])  # 测试集x，每一份每轮选一次
    test_y = list(len(texts_1[k]) * [1]) + list(len(texts_0[k]) * [-1])  # 测试集y，正例为1，负例为0

    index_rest = sorted(set(range(10)) - set([k]))  # 除去测试集剩余索引
    index_p = random.sample(index_rest, percent_p)  # 随机选择percent_p个p索引
    texts_1 = np.array(texts_1)  # 转换为array格式可进行多维索引
    p = texts_1[index_p]  # p集合

    index_except_p = sorted(set(range(10)) - set([k]) - set(index_p))  # 除去测试集和p集合剩余索引
    texts_0 = np.array(texts_0)  # 转换为array格式可进行多维索引
    u = list(texts_1[index_except_p]) + list(texts_0[index_rest])  # u集合

    p = np.array([j for i in p for j in i])
    u = np.array([j for i in u for j in i])
    X = np.concatenate((p, u), axis=0)
    y = [1] * len(p) + [-1] * len(u)

    with open(r'C:\Users\yyveggie\Desktop\python\PU_Learning\RESVM\data\train_{}.libsvm'.format(str(k)), 'w') as f:
        for i in range(len(X)):
            f.write(str(y[i]))
            f.write(' ')
            f.write(str(X[i]).replace(' ', '').replace('{', '').replace('}', '').replace(',', ' '))
            f.write('\n')
    with open(r'C:\Users\yyveggie\Desktop\python\PU_Learning\RESVM\data\test_{}.libsvm'.format(str(k)), 'w') as f:
        for o in range(len(test)):
            f.write(str(test_y[o]))
            f.write(' ')
            f.write(str(test[o]).replace(' ', '').replace('{', '').replace('}', '').replace(',', ' '))
            f.write('\n')

## 然后去使用cygwin调用emsemblesvm，用来训练数据集和预测样本

## 最后使用Perfoemance.py用来计算F1-Score和Accuracy