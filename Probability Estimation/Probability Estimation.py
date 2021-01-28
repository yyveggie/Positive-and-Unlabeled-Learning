''' CSV文件格式 '''
import random
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import time
from Conversion import CSV

''' Create data '''

e1_f1_binary = []
e1_accuracy = []
e2_f1_binary = []
e2_accuracy = []
e3_f1_binary = []
e3_accuracy = []
path = r'C:\Users\yyveggie\Desktop\UCI\Conversion\mushroom.csv'

texts_1, texts_0 = CSV(path)

def SplitDate(k, texts_1, texts_0):
    percent_p = 3  # p所占份数
    texts_1 = np.array_split(texts_1, 10)  # 将类别为1的样本集分成十份
    texts_0 = np.array_split(texts_0, 10)  # 将类别为0的样本集分成十份
    test = pd.DataFrame(list(texts_1[k]) + list(texts_0[k]))  # 测试集x，每一份每轮选一次
    y = pd.Series(list(len(texts_1[k]) * [1]) + list(len(texts_0[k]) * [0]))  # 测试集y，正例为1，负例为0

    index_rest = sorted(set(range(10)) - {k})  # 除去测试集剩余索引
    index_p = random.sample(index_rest, percent_p)  # 随机选择percent_p个p索引
    texts_1 = np.array(texts_1)  # 转换为array格式可进行多维索引
    p = texts_1[index_p]  # p集合

    index_except_p = sorted(set(range(10)) - {k} - set(index_p))  # 除去测试集和p集合剩余索引
    texts_0 = np.array(texts_0)  # 转换为array格式可进行多维索引
    u = list(texts_1[index_except_p]) + list(texts_0[index_rest])  # u集合
    p = pd.DataFrame(np.array([j for i in p for j in i]))
    u = pd.DataFrame(np.array([j for i in u for j in i]))
    p['y'] = 1
    u['y'] = pd.Series([1] * len([j for i in list(texts_1[index_except_p]) for j in i]) + [0] * len([j for i in list(texts_0[index_rest]) for j in i]))

    test['y'] = y
    return u, p, test

for k in range(10):
    u, p, test = SplitDate(k, texts_1, texts_0)

    # 普通分类器，普通数据示例
    classifier = SVC

    # 额外产生一些undefined data
    p["s"] = p["y"] * np.random.randint(1, 2, len(p))      # 只有positive会被labeled，所以s=1时，y必定等于1，当s=0时，y=0或者y=1
    u["s"] = u["y"] * np.random.randint(0, 1, len(u))

    # 数据切割
    y_train = p[['y', 's']].append(u[['y', 's']])                                   # 先将y标签提出
    X_train = p.drop(['y', 's'], axis=1).append(u.drop(['y', 's'], axis=1))         # 组合训练集
    y_test = test
    X_test = test.drop(['y'], axis=1)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.33, random_state=42)    # 将训练集分为训练集和验证集

    # 拟合g分类器（用来返回一个样本被计算为labeled的概率，g(x) = p(s = 1|x)），g分类器被称作nontraditional classifier（非传统分类器）
    CL_g = classifier(probability=True, gamma='auto')
    CL_g.fit(X_train, y_train["s"])                         # 训练分类器，标签选择训练集s列

    # Estimating c
    X_val_p = X_val.loc[X_val.index[y_val["s"] == 1]]       # s=1，即验证集中标签为s=1的样本集
    d_p = CL_g.predict_proba(X_val_p)[:, 1]                 # 预测X_val_p
    e1 = d_p.sum()/len(X_val_p)
    print("e1：", e1)

    d_v = CL_g.predict_proba(X_val)[:, 1]                   # 预测整个验证集
    e2 = d_p.sum()/d_v.sum()
    print("e2：", e2)

    e3 = d_v.max()
    print("e3：", e3)

    # First way to create f classificator
    CL_f = classifier(probability=True, gamma='auto')
    CL_f.fit(X_train, y_train["s"])                 # 训练分类器g，标签选择训练集s列
    c = e1                                          # e2 or e3，选择e1是因为文章中说明e1的效果最佳
    print("当前c值为：", c)
    y_pred = np.array(CL_f.predict_proba(X_test)[:, 1]/c > 0.5, dtype=int)      # g分类器预测结果/c = f分类器预测结果
    print("当前c值下f分类器准确率为：", accuracy_score(y_test["y"], y_pred))

    # Second way to improve classifier f

    # The way, that was used by the authors of article is to create weights
    # of each sample build on g prediction for value and parameter c
    # 根据文章中的方法，根据传统分类器g(x)就可以将每个样本加上一个权重，
    # 此时可以用带样本权重的分类器进行进一步的分类，可以取得更好的效果
    def create_new_train(e, X_train, y_train, CL):
        X_train_new = pd.DataFrame(X_train.loc[y_train.index[y_train["s"] == 1]])   # 标记样本
        X_train_new["w"] = 1    # 标记样本权重为1
        y_train_new = pd.DataFrame([1] * len(X_train_new), columns=["s"])       # 标记样本标签s
        X_train_1 = pd.DataFrame(X_train.loc[y_train.index[y_train["s"] == 0]])     # 未标记样本
        g_x = CL.predict_proba(X_train_1)[:, 1]
        X_train_1["w"] = (1 - e) / e * g_x / (1 - g_x)
        y_train_1 = pd.DataFrame([1] * len(X_train_1), columns=["s"])
        X_train_2 = pd.DataFrame(X_train.loc[y_train.index[y_train["s"] == 0]])
        g_x = CL.predict_proba(X_train_2)[:, 1]
        X_train_2["w"] = 1 - (1 - e) / e * g_x / (1 - g_x)
        y_train_2 = pd.DataFrame([0] * len(X_train_2), columns=["s"])
        X_train_new = X_train_new.append(X_train_1, ignore_index=True, sort=False)
        X_train_new = X_train_new.append(X_train_2, ignore_index=True, sort=False)
        y_train_new = y_train_new.append(y_train_1, ignore_index=True, sort=False)
        y_train_new = y_train_new.append(y_train_2, ignore_index=True, sort=False)
        return X_train_new, y_train_new

    # e1
    CL_f = classifier(probability=True, gamma='auto')
    X_train_new, y_train_new = create_new_train(e1, X_train, y_train, CL_g)
    X_new = X_train_new.drop(['w'], axis=1)
    CL_f.fit(X_new, y_train_new["s"], sample_weight=X_train_new['w'])
    y_true = y_test["y"]
    y_pred = CL_f.predict(X_test)

    e1_f1_binary.append(f1_score(y_true, y_pred, average='binary'))
    e1_accuracy.append(accuracy_score(y_true, y_pred))

    # e2
    CL_f = classifier(probability=True, gamma='auto')
    X_train_new, y_train_new = create_new_train(e2, X_train, y_train, CL_g)
    X_new = X_train_new.drop(['w'], axis=1)
    CL_f.fit(X_new, y_train_new["s"], sample_weight=X_train_new['w'])
    y_true = y_test["y"]
    y_pred = CL_f.predict(X_test)

    e2_f1_binary.append(f1_score(y_true, y_pred, average='binary'))
    e2_accuracy.append(accuracy_score(y_true, y_pred))

    # e3
    CL_f = classifier(probability=True, gamma='auto')
    X_train_new, y_train_new = create_new_train(e3, X_train, y_train, CL_g)
    X_new = X_train_new.drop(['w'], axis=1)
    CL_f.fit(X_new, y_train_new["s"], sample_weight=X_train_new['w'])
    y_true = y_test["y"]
    y_pred = CL_f.predict(X_test)

    e3_f1_binary.append(f1_score(y_true, y_pred, average='binary'))
    e3_accuracy.append(accuracy_score(y_true, y_pred))

print("F1-Score for e1：", np.mean(np.array(e1_f1_binary)))
print("Accuracy for e1：", np.mean(np.array(e1_accuracy)))
print("*" * 50)
print("F1-Score for e2：", np.mean(np.array(e2_f1_binary)))
print("Accuracy for e2：", np.mean(np.array(e2_accuracy)))
print("*" * 50)
print("F1-Score for e3：", np.mean(np.array(e3_f1_binary)))
print("Accuracy for e3：", np.mean(np.array(e3_accuracy)))