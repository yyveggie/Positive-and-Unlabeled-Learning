import numpy as np
import pandas as pd
from scipy.stats import norm, laplace
import torch.nn as nn
import torch.optim as optim
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import pickle

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

from IPython import display

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

import random
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score, train_test_split
from scipy.stats import gaussian_kde
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score, log_loss, mean_absolute_error, mean_squared_error, brier_score_loss
from sklearn.metrics import precision_score, recall_score, roc_auc_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.svm import OneClassSVM

from scipy.stats import linregress
from scipy.optimize import minimize
from scipy.stats import t
from statsmodels.stats.multitest import multipletests

from keras.layers import Dense, Dropout, Input, BatchNormalization
from keras.models import Model, Sequential
from keras.optimizers import Adam
from keras.losses import binary_crossentropy
from keras.callbacks import EarlyStopping

from algorithms import *
from utils import *
from KMPE import *
from NN_functions import *

from torchvision.datasets import MNIST

from tqdm import tqdm_notebook as tqdm
# from tqdm import tqdm as tqdm

import warnings
warnings.filterwarnings('ignore')

def read_data(data_mode, truncate=None, random_state=None):
    if data_mode == 'bank':
        df = pd.read_csv('UCI//bank//bank-full.csv', sep=';')
        df['balance'] = normalize_col(df['balance'])
        df = dummy_encode(df)
        df.rename(columns={'y': 'target'}, inplace=True)

    elif data_mode == 'concrete':
        df = pd.read_excel('UCI//concrete//Concrete_Data.xls')
        df = normalize_cols(df)
        df.rename(columns={'Concrete compressive strength(MPa, megapascals) ': 'target'}, inplace=True)
        df['target'] = reg_to_class(df['target'])

    elif data_mode == 'housing':
        df = pd.read_fwf('UCI//housing//housing.data.txt', header=None)
        df = normalize_cols(df)
        df.rename(columns={13: 'target'}, inplace=True)
        df['target'] = reg_to_class(df['target'])

    elif data_mode == 'landsat':
        df = pd.read_csv('UCI//landsat//sat.trn.txt', header=None, sep=' ')
        df = pd.concat([df, pd.read_csv('UCI//landsat//sat.tst.txt', header=None, sep=' ')])
        df = normalize_cols(df, columns=[x for x in range(36)])
        df.rename(columns={36: 'target'}, inplace=True)
        df['target'] = mul_to_bin(df['target'])

    elif data_mode == 'mushroom':
        df = pd.read_csv('UCI//mushroom//agaricus-lepiota.data.txt', header=None)
        df = dummy_encode(df)   # Auto encodes any dataframe column of type category or object
        df.rename(columns={0: 'target'}, inplace=True)  # 将真实标签列列名改为 "target"

    elif data_mode == 'pageblock':
        df = pd.read_fwf('UCI//pageblock//page-blocks.data', header=None)
        df = normalize_cols(df, columns=[x for x in range(10)])
        df.rename(columns={10: 'target'}, inplace=True)
        df['target'] = mul_to_bin(df['target'], 1)

    elif data_mode == 'shuttle':
        df = pd.read_csv('UCI//shuttle//shuttle.trn', header=None, sep=' ')
        df = pd.concat([df, pd.read_csv('UCI//shuttle//shuttle.tst.txt', header=None, sep=' ')])
        df = normalize_cols(df, columns=[x for x in range(9)])
        df.rename(columns={9: 'target'}, inplace=True)
        df['target'] = mul_to_bin(df['target'], 1)

    elif data_mode == 'spambase':
        df = pd.read_csv('UCI//spambase//spambase.data.txt', header=None, sep=',')
        df = normalize_cols(df, columns=[x for x in range(57)])
        df.rename(columns={57: 'target'}, inplace=True)

    elif data_mode == 'wine':
        df = pd.read_csv('UCI//wine//winequality-red.csv', sep=';')
        df_w = pd.read_csv('UCI//wine//winequality-white.csv', sep=';')
        df['target'] = 1
        df_w['target'] = 0
        df = pd.concat([df, df_w])
        df = normalize_cols(df, [x for x in df.columns if x != 'target'])

    elif data_mode.startswith('mnist'):
        data = MNIST('mnist', download=True, train=True)
        data_test = MNIST('mnist', download=True, train=False)

        df = data.train_data
        target = data.train_labels
        df_test = data_test.test_data
        target_test = data_test.test_labels

        df = pd.DataFrame(torch.flatten(df, start_dim=1).detach().numpy())
        df_test = pd.DataFrame(torch.flatten(df_test, start_dim=1).detach().numpy())
        df = pd.concat([df, df_test])
        df = normalize_cols(df)

        target = pd.Series(target.detach().numpy())
        target_test = pd.Series(target_test.detach().numpy())
        target = pd.concat([target, target_test])

        if data_mode == 'mnist_1':
            target[target % 2 == 0] = 0
            target[target != 0] = 1
        elif data_mode == 'mnist_2':
            target[target < 5] = 0
            target[target >= 5] = 1
        elif data_mode == 'mnist_3':
            target[target.isin({0, 3, 5, 6, 7})] = 0
            target[target.isin({1, 2, 4, 8, 9})] = 1

        df['target'] = target

    elif data_mode.startswith('cifar10'):

        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                download=True, transform=transform)
        #         trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
        #                                                   shuffle=True, num_workers=2)

        data = trainset.data
        target = trainset.targets
        #         if truncate is not None and truncate < trainset.data.shape[0]:
        #             np.random.seed(random_state)
        #             mask = np.random.choice(np.arange(trainset.data.shape[0]), truncate, replace=False)
        #             np.random.seed(None)
        #             data = trainset.data[mask]
        #             target = trainset.targets[mask]
        data = data / 128 - 1

        classes = ('plane', 'car', 'bird', 'cat',
                   'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        target = pd.Series(target)
        target[target.isin([0, 1, 8, 9])] = 1
        target[target != 1] = 0

        df = pd.DataFrame(data.reshape(data.shape[0], -1))
        df['target'] = target

    # 1 = N, 0 = P
    df['target'] = 1 - df['target']

    if truncate is not None and truncate < df.shape[0]:
        if truncate > 1:
            df = df.sample(n=truncate, random_state=random_state)
        elif truncate > 0:
            df = df.sample(frac=truncate, random_state=random_state)
    return df

shapes = {'bank': {0.95: (1000, int(39922 / 19), 39922),
                   0.75: (1000, 4289, 4289*3),
                   0.50: (1000, 4289, 4289),
                   0.25: (1000, 4289, int(4289 / 3)),
                   0.05: (1000, 4289, int(4289 / 19))},
          'concrete': {0.95: (100, int(540 / 19), 540),
                   0.75: (100, 180, 540),
                   0.50: (100, 390, 390),
                   0.25: (100, 390, 130),
                   0.05: (100, 380, 20)},
          'housing': {0.95: (194, 15, 297),
                   0.75: (110, 99, 297),
                   0.50: (50, 159, 159),
                   0.25: (50, 159, 53),
                   0.05: (57, 152, 8)},
          'landsat': {0.95: (1000, 189, 3594),
                   0.75: (1000, 1000, 3000),
                   0.50: (1000, 1841, 1841),
                   0.25: (1000, 1841, int(1841 / 3)),
                   0.05: (1000, 1841, int(1841 / 19))},
          'mushroom': {0.95: (1000, 200, 3800),
                   0.75: (1000, 1000, 3000),
                   0.50: (1000, 2000, 2000),
                   0.25: (1000, 2916, 972),
                   0.05: (990, 2926, 154)},
          'pageblock': {0.95: (100, 234, 4446),
                   0.75: (100, 460, 1380),
                   0.50: (100, 460, 460),
                   0.25: (101, 459, 153),
                   0.05: (104, 456, 24)},
          'shuttle': {0.95: (1000, int(45586 / 19), 45586),
                   0.75: (1000, 11414, 11414 * 3),
                   0.50: (1000, 11414, 11414),
                   0.25: (1000, 11414, int(11414 / 3)),
                   0.05: (1000, 11414, int(11414 / 19))},
          'spambase': {0.95: (400, 147, 2788),
                   0.75: (400, 929, 2788),
                   0.50: (400, 1413, 1413),
                   0.25: (400, 1413, 471),
                   0.05: (407, 1406, 74)},
          'wine': {0.95: (500, int(4898 / 19), 4898),
                   0.75: (500, 1099, 1099 * 3),
                   0.50: (500, 1099, 1099),
                   0.25: (500, 1099, int(1099 / 3)),
                   0.05: (500, 1099, int(1099 / 19))},
          'mnist_1': {0.95: (1000, int(34418 / 19), 34418),
                   0.75: (1000, int(34418 / 3), 34418),
                   0.50: (1000, 34418, 34418),
                   0.25: (1000, 34582, int(34582 / 3)),
                   0.05: (1000, 34582, int(34582 / 19))},
          'cifar10': {0.95: (1000, int(30000 / 19), 30000),
                   0.75: (1000, 10000, 30000),
                   0.50: (1000, 19000, 19000),
                   0.25: (1000, 19000, int(19000 / 3)),
                   0.05: (1000, 19000, int(19000 / 19))}}

LRS = {
    'bank': 5e-4,
    'concrete': 1e-4,
    'housing': 1e-4,
    'landsat': 1e-5,
    'mushroom': 1e-4,
    'pageblock': 1e-4,
    'shuttle': 1e-4,
    'spambase': 1e-5,
    'wine': 5e-5,
    'mnist_1': 1e-4,
    'mnist_2': 1e-4,
    'mnist_3': 1e-4,
    'cifar10': 1e-4,
}

def make_pu(df, data_mode, alpha, random_state=None):
    # α表示p在u中的比例
    df['target_pu'] = df['target']  # 创造pu标签列"target_pu"，初始化与原标签列"target"的标签相同
    n_pos, n_pos_to_mix, n_neg_to_mix = shapes[data_mode][1 - alpha]    # 在当前α下，读取p、u中的p、u中的n的数量，其中1 - α的意思是n在u中的比例
    df_pos = df[df['target'] == 0].sample(n=n_pos+n_pos_to_mix, random_state=random_state, replace=False).reset_index(drop=True)    # 从df中找到指定数量的正例，正例标签为0
    df_neg = df[df['target'] == 1].sample(n=n_neg_to_mix, random_state=random_state, replace=False).reset_index(drop=True)  # 从df中找到指定数量的负例，负例标签为1
    df_pos.loc[df_pos.sample(n=n_pos_to_mix, random_state=random_state, replace=False).index, 'target_pu'] = 1  # 对于部分正例按照p在u中的数量标记为负例
    return pd.concat([df_pos, df_neg]).sample(frac=1).reset_index(drop=True)    # 合并df_pos和df_neg，sample(frac=1)表示随机排列，reset_index(drop=True) 表示索引重置



''' DEDPUL on specific single data sets '''

# data_mode = 'bank' # 0.11, (5289, 39922, 45211)
# data_mode = 'concrete' # 0.47, (490, 540, 1030)
# data_mode = 'housing' # 0.41, (209, 297, 506)
# data_mode = 'landsat' # 0.44, (2841, 3594, 6435)
data_mode = 'mushroom' # 0.48, (3916, 4208, 8124)
# data_mode = 'pageblock' # 0.1, (560, 4913, 5473)
# data_mode = 'shuttle' # 0.21, (12414, 45586, 58000)
# data_mode = 'spambase' # 0.39, (1813, 2788, 4601)
# data_mode = 'wine' # 0.24, (1599, 4898, 6497)
# data_mode = 'mnist_1' # 0.51, (35582, 34418, 70000)
# data_mode = 'cifar10' # 0.4, (20000, 30000, 50000)

alpha = 0.25    # alpha表示u中p的比例
df = read_data(data_mode, truncate=None)    # 读取原始数据
df = make_pu(df, data_mode, alpha=alpha, random_state=None)    # pu数据
alpha = 1 - ((df['target'] == 0).sum() / (df['target_pu'] == 1).sum()).item()   # alpha是正例在u中所占的比例
gamma = (df['target_pu'] == 1).sum() / df.shape[0]  # gamma是u在全部样本中的比例

print(df.shape)
print('alpha =', alpha)

data = df.drop(['target', 'target_pu'], axis=1).values  # 将样本剔除标签值，只留下特征值
all_conv = False
if data_mode == 'cifar10':
    data = np.swapaxes(data.reshape(data.shape[0], 32, 32, 3), 1, 3)
    all_conv = True
target = df['target_pu'].values  # 保留pu标签（非真实标签）

preds = estimate_preds_cv(data, target, cv=3, bayes=False, random_state=42, all_conv=all_conv,
                          n_networks=1, hid_dim=512, n_hid_layers=1, lr=LRS[data_mode], l2=1e-4, bn=True,
                          train_nn_options={'n_epochs': 500, 'bayes_weight': 1e-5,
                                            'batch_size': 64, 'n_batches': None, 'n_early_stop': 7,
                                            'disp': True, 'loss_function': 'log',
                                            'metric': roc_auc_loss, 'stop_by_metric': False})

# preds = estimate_preds_cv_catboost(
#     data, target, cv=5, n_networks=1, random_state=42, n_early_stop=20, verbose=True,
#     catboost_params={
#         'iterations': 500, 'depth': 8, 'learning_rate': 0.02, 'l2_leaf_reg': 10,
#     },
# )

print('ac', accuracy_score(df['target_pu'], preds.round()))
# print('bac', balanced_accuracy_score(df['target_pu'], preds.round()))
print('roc', roc_auc_score(df['target_pu'], preds))
print('brier', brier_score_loss(df['target_pu'], preds))