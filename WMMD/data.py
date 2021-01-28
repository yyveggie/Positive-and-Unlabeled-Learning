#!/usr/bin/env python
# coding: utf-8

import numpy as np
import random


def SplitData(k, texts_1, texts_0):
    # Generate training dataset for PU
    # Result: in dictionary {X, PU_labels, PN_labels} for experimental purpose.
    # X: independent variables
    # PU_labels: 0 for unlabled and 1 for labeled
    # PN_labels: -1 for true negative and 1 for true positive

    # Generate test set
    # Result: in dictionary {X, PN_labels}
    # X: independent variables
    # PN_labels: -1 for true negative and 1 for true positive

    percent_p = 3
    texts_1 = np.array_split(texts_1, 10)                                      # 将类别为1的样本集分成十份
    texts_0 = np.array_split(texts_0, 10)                                      # 将类别为0的样本集分成十份
    X_te = np.array(list(texts_1[k]) + list(texts_0[k]))                               # 测试集x，每一份每轮选一次
    Y_te = np.array(list(len(texts_1[k])*[1]) + list(len(texts_0[k])*[-1]))         # 测试集y，正例为1，负例为0
    te = np.hstack((X_te, Y_te.reshape(-1, 1))).astype('float32')
    # shuffle row only
    np.random.shuffle(te)

    index_rest = sorted(set(range(10)) - {k})                             # 除去测试集剩余索引
    index_p = random.sample(index_rest, percent_p)                             # 随机选择percent_p个p索引
    texts_1 = np.array(texts_1)                                                # 转换为array格式可进行多维索引
    p = texts_1[index_p]                                                       # p集合

    index_except_p = sorted(set(range(10)) - {k} - set(index_p))          # 除去测试集和p集合剩余索引
    texts_0 = np.array(texts_0)                                                # 转换为array格式可进行多维索引
    u = list(texts_1[index_except_p]) + list(texts_0[index_rest])              # u集合
    Y_u = np.array([1]*len([j for i in list(texts_1[index_except_p]) for j in i]) +
                   [-1]*len([j for i in list(texts_0[index_rest]) for j in i]))
    X_p_tr = np.array([j for i in p for j in i])
    X_u_tr = np.array([j for i in u for j in i])
    PU_label = np.hstack((np.ones(len(X_p_tr)), np.zeros(len(X_u_tr))))
    PN_label = np.hstack((np.ones(len(X_p_tr)), Y_u))
    X_PU_tr = np.vstack((X_p_tr, X_u_tr))
    PU_tr = np.hstack((X_PU_tr, PU_label.reshape(-1, 1), PN_label.reshape(-1, 1))).astype('float32')
    # shuffle row only
    np.random.shuffle(PU_tr)

    return {'X': PU_tr[:, :-2], 'PU_label': PU_tr[:, -2], 'PN_label': PU_tr[:, -1]}, \
           {'X': te[:, :-1], 'PN_label': te[:, -1]}


def generate_normal_PU_tr_data(pi_plus, n_p, n_u, **kwargs):
    #Generate normal training dataset for PU
    #Result: in dictionary {X, PU_labels, PN_labels} for experimental purpose.
    #X: independent variables
    #PU_labels: 0 for unlabled and 1 for labeled
    #PN_labels: -1 for true negative and 1 for true positive

    # Parameters to generate
    mu_p = kwargs.get('mu_p', np.array([1, 1])/np.sqrt(2))
    mu_n = kwargs.get('mu_n', np.array([-1, -1])/np.sqrt(2))
    cov_p = kwargs.get('cov_p', np.eye(2))
    cov_n = kwargs.get('cov_n', np.eye(2))
    
    # Generate dataset
    X_p_tr = np.random.multivariate_normal(mu_p, cov_p, n_p)
    Y_u = np.sort(np.random.choice([-1,1], n_u, p=[1-pi_plus, pi_plus]))
    X_u_tr = np.vstack((np.random.multivariate_normal(mu_n, cov_n, np.count_nonzero(Y_u==-1)),
                        np.random.multivariate_normal(mu_p, cov_p, np.count_nonzero(Y_u==1)))
                      )
    PU_label = np.hstack((np.ones(n_p),np.zeros(n_u)))
    PN_label = np.hstack((np.ones(n_p),Y_u))
    X_PU_tr = np.vstack((X_p_tr, X_u_tr))
    PU_tr = np.hstack((X_PU_tr, PU_label.reshape(-1,1), PN_label.reshape(-1,1))).astype('float32')
    #shuffle row only
    np.random.shuffle(PU_tr)
    #make it to dictionary
    return {'X': PU_tr[:, :-2], 'PU_label': PU_tr[:, -2], 'PN_label': PU_tr[:,-1]}


def generate_normal_te_data(pi_plus, n_te, **kwargs):
    #Generate normal test set
    #Result: in dictionary {X, PN_labels}
    #X: independent variables
    #PN_labels: -1 for true negative and 1 for true positive
    
    # Parameters to generate
    mu_p = kwargs.get('mu_p', np.array([1,1])/np.sqrt(2) )
    mu_n = kwargs.get('mu_n', np.array([-1,-1])/np.sqrt(2) )
    cov_p = kwargs.get('cov_p', np.eye(2) )
    cov_n = kwargs.get('cov_n', np.eye(2) )

    # Generate datasets
    Y_te = np.sort(np.random.choice([-1,1], n_te, p=[1-pi_plus,pi_plus]))
    X_te = np.concatenate((np.random.multivariate_normal(mu_n, cov_n, np.count_nonzero(Y_te==-1)),
                              np.random.multivariate_normal(mu_p, cov_p, np.count_nonzero(Y_te==1))),
                             axis=0)
    te = np.hstack((X_te, Y_te.reshape(-1,1))).astype('float32')
    
    #shuffle row only
    np.random.shuffle(te) 
    
    #make it to dictionary
    return {'X':te[:,:-1], 'PN_label':te[:,-1]}