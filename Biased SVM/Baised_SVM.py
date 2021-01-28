from sklearn.svm import SVC
import numpy as np
import random
from Conversion import CSV
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

svc = SVC(C=1.0, kernel='rbf', gamma='auto', probability=True, random_state=2018)

f1 = []
accuracy = []
path = r'C:\Users\yyveggie\Desktop\UCI\Conversion\mushroom.csv'
C_p = 0.05
C_n = 0.3

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


def biased_svm(cost_fp, cost_fn, P, U, test_x, svm=svc, return_proba=False):
    assert cost_fn > cost_fp > 0, '对FN应赋予更高的代价'
    X_train = np.r_[P, U]
    y_train = np.r_[np.ones(len(P)), np.zeros(len(U))]
    weight = [cost_fn if i else cost_fp for i in y_train]
    svm.fit(X_train, y_train, sample_weight=weight)
    y_pred = svm.predict(test_x)
    if return_proba:
        y_prob = svm.predict_proba(U)[:, -1]
        return y_pred, y_prob
    else:
        return y_pred


for k in range(10):
    print("Iter：", k)
    p, u, test_x, test_y = SplitData(k, texts_1, texts_0)
    y_pred = biased_svm(C_p, C_n, p, u, test_x)
    print(y_pred)
    f1.append(f1_score(test_y, y_pred, average='binary'))
    accuracy.append(accuracy_score(test_y, y_pred))

print("F1-Score：", np.mean(np.array(f1)))
print("Accuracy：", np.mean(np.array(accuracy)))