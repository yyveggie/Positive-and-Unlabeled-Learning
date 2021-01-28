'''CSV文件格式'''

import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from sklearn.naive_bayes import GaussianNB
import random
import time
from Conversion import CSV

f1_binary = []
accuracy = []
s = time.time()
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

### Two-Step ###

for k in range(10):
    print("iter", k)
    p, u, test_x, test_y = SplitData(k, texts_1, texts_0)
    p = pd.DataFrame(p)
    u = pd.DataFrame(u)
    # 首先将所有的U数据当做负样本，P当做正样本作为训练集训练分类器
    X_train = pd.concat([p, u])
    y_train = pd.Series([1] * len(p) + [0] * len(u))
    rf = GaussianNB()
    rf.fit(X_train, y_train)

    # 对U中的样本进行预测，这样U中每个样本都会更新标签
    pred = rf.predict(X_train)

    # STEP 1

    # Classifier to be used for step 2
    rf2 = GaussianNB()

    num_p_2 = len(X_train)                # 将停止条件的最初值设定为一个很大的数，防止第一轮就停止

    for i in range(1000):
        num_p = list(pred).count(1)       # 当前识别的个数

        if num_p < num_p_2:
            num_p_2 = num_p                                                # 当前识别的个数算作上一轮识别的个数
            stand = pd.Series([n == 0 for n in pred])                      # 找到那些依旧在U中被判定为负的（n == 0）的样本
            U = pd.DataFrame(X_train.values[stand])                        # U中保留那些依旧判定为负的（n == 0）样本
            print("还有 %d" % len(U) + '的U')
            print('Step 1 labeled %d new positives and %d new negatives.' % (len([k for k in stand if k == 1]), len([o for o in stand if o == 0])))
            print('Doing step 2 ... ')

            # STEP 2
            U_y = len(U) * [0]
            X = pd.DataFrame(p).append(pd.DataFrame(U))
            X_y = pd.Series(len(p) * [1] + U_y)

            rf2 = GaussianNB()                  # 在原本P和新U上重新训练
            rf2.fit(X, X_y)

            pred = rf2.predict(U)               # 得到的分类器重新对新U进行预测
            num_p = list(pred).count(1)         # 统计里面被分类为P的个数
            X_train = U

        else:
            y_pred = rf2.predict(test_x)        # 停止条件终止之后，返回最后一个分类器，并对测试集分类
            f1_binary.append(f1_score(test_y, y_pred, average='binary'))
            accuracy.append(accuracy_score(test_y, y_pred))
            break

print("F1-Score：", np.mean(np.array(f1_binary)))
print("Accuracy：", np.mean(np.array(accuracy)))
print("Time：", time.time() - s)