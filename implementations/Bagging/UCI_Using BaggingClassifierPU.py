from __future__ import division, print_function
'''Source: https://github.com/phuijse/bagging_pu/blob/master/PU_Learning_simple_example.ipynb'''
'''Reproduces: A bagging SVM to learn from positive and unlabeled examples'''
'''Dataset: UCI'''
'''Preprocessing follows: Positive and unlabeled learning in categorical data'''

# raises

import numpy as np
from collections import Counter
import pandas as pd
import random
from sklearn.tree import DecisionTreeClassifier
from baggingPU import BaggingClassifierPU
import matplotlib.pyplot as plt


''' Create data '''


texts = pd.read_csv(r'data/dermatology.data', header=None)
labels = dict(Counter(texts[34]))
sorted_labels = sorted(labels.items(), key=lambda x: x[1], reverse=True)        # sort by class size and keep the two largest classes
labels_1 = sorted_labels[0][0]                      # the largest class
labels_2 = sorted_labels[1][0]                      # the second largest class

X = texts                  # the full sample set
indexs = [i for i, a in enumerate(texts[34]) if a == labels_2]       # indices of the labeled samples
unindexs = [j for j, b in enumerate(texts[34]) if b == labels_1]     # indices of the unlabeled samples
labeled_texts = X.iloc[indexs]       # the labeled samples
unlabeled_texts = X.iloc[unindexs]   # the unlabeled samples
print("initial number of labeled samples:", len(labeled_texts))
print("initial number of unlabeled samples:", len(unlabeled_texts))

num_pos = 30                               # number of positives, as in the paper
percent_pos = 0.3                          # share of the positives assigned to P
num_p = round(num_pos * percent_pos)       # size of P
# num_neg = 136 - (num_pos - num_p)          # number of negatives; the count in the paper cannot be reproduced here
num_neg = 100
num_test = 19                              # size of the test set
index_pos = random.sample(range(len(labeled_texts)), num_pos)       # indices of the positive samples
pos = labeled_texts.iloc[index_pos]                                 # the positive samples
index_neg = random.sample(range(len(unlabeled_texts)), num_neg)     # indices of the negative samples
neg = unlabeled_texts.iloc[index_neg]                               # the negative samples

T = labeled_texts.iloc[sorted(set(range(len(labeled_texts))) - set(index_pos))].append(
    unlabeled_texts.iloc[sorted(set(range(len(unlabeled_texts))) - set(index_neg))])    # what remains after removing the positive and negative samples
index_test = random.sample(range(len(T)), num_test)
test = T.iloc[index_test]

index_p = random.sample(range(len(pos)), num_p)     # indices of the P samples
p = pos.iloc[index_p]                       # the P samples
index_up = sorted(set(range(len(pos))) - set(index_p))      # indices of the positives left after removing P
up = pos.iloc[index_up]                     # the positives left after removing P
u = up.append(neg)                          # the U samples
u = u.drop([34], axis=1)                    # drop the label column
p = p.drop([34], axis=1)                    # drop the label column
test.loc[test[34] == 1, 34] = 0          # relabel U from 1 to 0
test.loc[test[34] == 3, 34] = 1          # relabel the positives from 3 to 1
y = pd.Series(test[34])                     # save the test labels
test = test.drop([34], axis=1)              # drop the label column
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

# store the score this method assigns
results = pd.DataFrame({
    'truth': y_orig,   # The true labels
    'label': y,        # The labels to be shown to models in experiment
}, columns=['truth', 'label'])
results['output_skb'] = bc.oob_decision_function_[:, 1]

# plot the result
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