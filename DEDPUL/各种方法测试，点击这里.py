import numpy as np
import pandas as pd
from scipy.stats import norm, laplace
import random
from sklearn.metrics import accuracy_score, log_loss, mean_absolute_error, mean_squared_error, confusion_matrix
from sklearn.metrics import precision_score, recall_score, roc_auc_score, brier_score_loss, f1_score
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from IPython import display
from algorithms import *
from utils import *
from KMPE import *
from NN_functions import *
import warnings
warnings.filterwarnings('ignore')

'''data'''

# specify distributions to sample data from.
# mode = 'normal'
mode ='laplace'

# feel free to play with parameters of distributions;
# initially we recommend to stick to cases of s1=s2

# centers and standard deviations of P and N distributions
mu1 = 0
s1 = 1
mu2 = 4
s2 = 1

# alpha is proportion of N in U; (1 - alpha) is proportion of P in U; these will be unknown for methods below;
# alpha表示N在U中的比例，对于以下方法这将是未知的
# note that not alpha but alpha^* (computed below) is the proportion that the methods are supposed to identify
# (find out why in the paper)
alpha = 0.75

if mode == 'normal':
    p1 = lambda x: norm.pdf(x, mu1, s1)
    p2 = lambda x: norm.pdf(x, mu2, s2)
    pm = lambda x: p1(x) * (1 - alpha) + p2(x) * alpha
elif mode == 'laplace':
    p1 = lambda x: laplace.pdf(x, mu1, s1)
    p2 = lambda x: laplace.pdf(x, mu2, s2)
    pm = lambda x: p1(x) * (1 - alpha) + p2(x) * alpha

# 具体分布可视化

# plt.plot([x/100 for x in range(-1000, 1000)], [p1(x/100) for x in range(-1000, 1000)], 'b')
# plt.plot([x/100 for x in range(-1000, 1000)], [p2(x/100) for x in range(-1000, 1000)], 'g')
# plt.plot([x/100 for x in range(-1000, 1000)], [pm(x/100) for x in range(-1000, 1000)], 'r')
#
# plt.legend(handles=(Line2D([], [], linestyle='-', color='b'),
#                     Line2D([], [], linestyle='-', color='g'),
#                     Line2D([], [], linestyle='-', color='r')),
#            labels=('$f_p(x)$', '$f_n(x)$', '$f_u(x)$'),
#            fontsize='x-large')
# plt.show()

# 根据正例分布fp(x)和未标签分布fu(x)生成p样本和u样本，但我们没有来自分布fn(x)的n样本

if mode == 'normal':
    sampler = np.random.normal
elif mode == 'laplace':
    sampler = np.random.laplace

mix_size = 2500
pos_size = 500

mix_data_test = np.append(sampler(mu1, s1, int(mix_size * (1 - alpha))),
                          sampler(mu2, s2, int(mix_size * alpha)))
pos_data_test = sampler(mu1, s1, int(pos_size))

data_test = np.append(mix_data_test, pos_data_test).reshape((-1, 1))
target_test = np.append(np.array([1] * mix_size), np.array([0] * pos_size))
target_test_true = np.append(np.array([0] * int(mix_size * (1 - alpha))), np.array([1] * int(mix_size * alpha)))
target_test_true = np.append(target_test_true, np.array([2] * pos_size))

mix_data_test = mix_data_test.reshape([-1, 1])
pos_data_test = pos_data_test.reshape([-1, 1])

data_test = np.concatenate((data_test, target_test.reshape(-1, 1), target_test_true.reshape(-1, 1)), axis=1)
np.random.shuffle(data_test)
target_test = data_test[:, 1]
target_test_true = data_test[:, 2]
data_test = data_test[:, 0].reshape(-1, 1)

print(mix_data_test)
print(type(mix_data_test))
print("*" * 100)
print(pos_data_test)
print(type(pos_data_test))
print("*" * 100)
print(data_test)
print(type(data_test))
print("*" * 100)
print(target_test)
print(type(target_test))
print("*" * 100)
print(target_test_true)
print(type(target_test_true))

# here we may estimate ground truth alpha^* for limited number of cases:
# laplace and normal distributions where either mean or std coincide.
# alpha^* is the desired proportion that the methods are supposed to identify.

cons_alpha = estimate_cons_alpha(mu2 - mu1, s2 / s1, alpha, mode)
print('alpha* =', cons_alpha)

# estimate y(x), the predictions of NTC

preds = estimate_preds_cv(data_test, target_test, n_networks=10,
                          train_nn_options={'n_epochs': 200, 'batch_size': 128,
                                            'n_batches': 20, 'n_early_stop': 3, 'disp': False, 'loss_function': 'log'})

print(target_test)
print(preds.round())
print('ac', accuracy_score(target_test, preds.round()))
print('roc', roc_auc_score(target_test, preds))

bw_mix = 0.05
bw_pos = 0.1
threshold = preds[target_test == 1].mean()
k_neighbours = 150
MT_coef = 0.25

# estimate density ratio g_p(y)/g_u(y) with tuned bandwidths, apply heuristics

diff = estimate_diff(preds, target_test,
                     bw_mix, bw_pos, 'logit', threshold, k_neighbours,
                     MT=True, MT_coef=MT_coef, decay_MT_coef=False)


# plt.plot(preds[target_test==1], diff, 'bo')

# Estimate priors alpha^* and posteriors f^*(p | x) with EM algorithm (if converges)
# or with alternative max_slope otherwise

test_alpha, poster = estimate_poster_em(diff, mode='dedpul', converge=True, nonconverge=True,
                                        max_diff=0.05, step=0.0025, alpha_as_mean_poster=True)#, alpha=cons_alpha)
print('alpha:', test_alpha, '\nmean_poster:', np.mean(poster), '\ncons_alpha:', cons_alpha)

plt.plot(preds[target_test == 1], poster, 'ro')
plt.ylabel('$f(p \mid x)$')
plt.xlabel('$y(x)$')