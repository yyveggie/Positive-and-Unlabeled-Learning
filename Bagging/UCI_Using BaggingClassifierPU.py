from __future__ import division, print_function
'''代码来源网站：https://github.com/phuijse/bagging_pu/blob/master/PU_Learning_simple_example.ipynb'''
'''文章复现：A bagging SVM to learn from positive and unlabeled examples'''
'''数据集：UCI'''
'''数据处理方式来源：Positive and unlabeled learning in categorical data'''

# 报错

import numpy as np
from collections import Counter
import pandas as pd
import random
from sklearn.tree import DecisionTreeClassifier
from baggingPU import BaggingClassifierPU
import matplotlib.pyplot as plt


''' Create data '''


texts = pd.read_csv(r'C:\Users\yyveggie\Desktop\UCI\dermatology.data', header=None)
labels = dict(Counter(texts[34]))
sorted_labels = sorted(labels.items(), key=lambda x: x[1], reverse=True)        # 按照样本数量排序，选择数量最多的两个类别
labels_1 = sorted_labels[0][0]                      # 样本数量最多的类别
labels_2 = sorted_labels[1][0]                      # 样本数量第二多的类别

X = texts                  # 总样本集
indexs = [i for i, a in enumerate(texts[34]) if a == labels_2]       # 已标记的索引
unindexs = [j for j, b in enumerate(texts[34]) if b == labels_1]     # 未标记的索引
labeled_texts = X.iloc[indexs]       # 已标记的样本集
unlabeled_texts = X.iloc[unindexs]   # 未标记的样本集
print("最初一共有已标记的样本集数目：", len(labeled_texts))
print("最初一共有未标记的样本集数目：", len(unlabeled_texts))

num_pos = 30                               # 正例样本数量（文中原文）
percent_pos = 0.3                          # p所占正例百分比
num_p = round(num_pos * percent_pos)       # p的数量
# num_neg = 136 - (num_pos - num_p)          # 负例样本数量（原文的数量在此处无法实现）
num_neg = 100
num_test = 19                              # 测试集数量
index_pos = random.sample(range(len(labeled_texts)), num_pos)       # 正例样本集索引
pos = labeled_texts.iloc[index_pos]                                 # 正例样本集
index_neg = random.sample(range(len(unlabeled_texts)), num_neg)     # 负例样本集索引
neg = unlabeled_texts.iloc[index_neg]                               # 负例样本集

T = labeled_texts.iloc[sorted(set(range(len(labeled_texts))) - set(index_pos))].append(
    unlabeled_texts.iloc[sorted(set(range(len(unlabeled_texts))) - set(index_neg))])    # 除去正例样本集和负例样本集之后的集合
index_test = random.sample(range(len(T)), num_test)
test = T.iloc[index_test]

index_p = random.sample(range(len(pos)), num_p)     # p样本集索引
p = pos.iloc[index_p]                       # p样本集
index_up = sorted(set(range(len(pos))) - set(index_p))      # 正例中除去p样本集后剩下的样本集索引
up = pos.iloc[index_up]                     # 正例中除去p样本集后剩下的样本集
u = up.append(neg)                          # u样本集
u = u.drop([34], axis=1)                    # 删除标签列
p = p.drop([34], axis=1)                    # 删除标签列
test.loc[test[34] == 1, 34] = 0          # 将原标签为1的U改为0
test.loc[test[34] == 3, 34] = 1          # 将原标签为3的正例改为1
y = pd.Series(test[34])                     # 保存测试集标签
test = test.drop([34], axis=1)              # 删除标签列
data_P = p.values
data_U = u.values
test = test.values


''' Transductive PU learning '''


X = pd.DataFrame(np.concatenate((data_P, data_U), axis=0))
y = pd.Series([1] * len(data_P) + [0] * len(data_U))
y_orig = pd.Series([1] * len(data_P) + [1] * len(up) + [0] * len(neg))
print(len(y))
print(len(y_orig))


bc = BaggingClassifierPU(
    DecisionTreeClassifier(),
    n_estimators=1000,
    max_samples=sum(y),     # Balance the positives and unlabeled in each bag
    n_jobs=-1               # Use all cores
)
bc.fit(X, y)

# 存储此方法分配的score
results = pd.DataFrame({
    'truth': y_orig,   # The true labels
    'label': y,        # The labels to be shown to models in experiment
}, columns=['truth', 'label'])
results['output_skb'] = bc.oob_decision_function_[:, 1]

# 可视化结果
plt.scatter(
    X[y == 0].feature1, X[y == 0].feature2,
    c=results[y == 0].output_skb,
    linewidth=0,
    s=50,
    alpha=0.5,
    cmap='jet_r'
)
plt.colorbar(label='Scores given to unlabeled points')
plt.title(r'Using ${\tt BaggingClassifierPU}$')
plt.show()