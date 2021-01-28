#!/usr/bin/env python
# coding: utf-8

import torch
from torch.utils.data import DataLoader

from data import *
from wmmd import WMMD
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('Agg')
from sklearn.datasets import make_moons, make_circles

def illustrate_decision_boundary_moons(pi_plus, n_p, n_u, device, gamma_list, batch_size,seed=None,
                                         outfile='contour_moons.pdf'):
    print('Plotting the illustration of WMMD decision boundary')
    print('##The training dataset:two_moons')
    print('##class-prior:{}'.format(pi_plus))
    print('##Positive sample size:{}'.format(n_p))
    print('##Unlabeled sample size:{}'.format(n_u))
    
    #Set seed for reproduciblility
    if seed != None: 
        np.random.seed(seed)

    #generate dataset for training and validation
    data_PU_tr = generate_moons_PU_tr_data(pi_plus, int(1.25*n_p), int(1.25*n_u))

    #For grid range
    xmin, xmax = -6, 10
    ymin, ymax = -4, 6
    
    #generate delta-grid points
    delta = 0.5
    x = np.arange(xmin, xmax, delta)
    y = np.arange(ymin, ymax, delta)
    X, Y = np.meshgrid(x, y)
    d1, d2 = X.shape
    Xre = X.reshape(-1,1)
    Yre = Y.reshape(-1,1)
    te = np.hstack((Xre,Yre)).astype('float32')

    #Training part
    hyper_wmmd_optimal = [0,0,10] # threshold, gamma, val            
    for gam in gamma_list:
        model_wmmd = WMMD(data_dict = data_PU_tr,
                                    device=device, pi_plus=pi_plus,
                                    dict_parameter={'gamma':gam})
        model_wmmd.calculate_val_risk()
        if hyper_wmmd_optimal[2] > model_wmmd.val_loss:
            hyper_wmmd_optimal[0] = model_wmmd.threshold
            hyper_wmmd_optimal[1] = gam
            hyper_wmmd_optimal[2] = model_wmmd.val_loss

    model_wmmd = WMMD(data_dict = data_PU_tr,
                      device=device, pi_plus=pi_plus,
                      dict_parameter={'gamma':hyper_wmmd_optimal[1]})

    #Evaluating the function values
    generator_te = DataLoader(te, batch_size=batch_size, shuffle=False)
    score_list_wmmd = []

    for te_batch in generator_te:
        te_batch = (te_batch).to(device)
        score_wmmd = model_wmmd.score(te_batch).data.cpu().numpy().ravel()
        score_list_wmmd.append(score_wmmd)


    '''
    Plot
    '''
    plt.figure(figsize=(5, 5))

    #True function plot
    generated_true = make_moons(1000)
    P_label = generated_true[1] == 1
    N_label = generated_true[1] == 0
    plt.scatter(generated_true[0][P_label,0]*4, generated_true[0][P_label,1]*4, alpha=0.2, marker='o', color='blue', s=2)
    plt.scatter(generated_true[0][N_label,0]*4, generated_true[0][N_label,1]*4, alpha=0.2, marker='o', color='red', s=2) 

    #scatter plot with train datasets only
    training_data = list(model_wmmd.generator_pu_tr)[0].numpy()
    #print('n_p:{}, n_u:{}'.format(model_wmmd.n_tr_p, model_wmmd.n_tr_u))
    train_ind = [np.where(data_PU_tr['X'] == training_data[i,0:2])[0][0] for i in range(training_data.shape[0])]
    data_PU_tr = {'X':data_PU_tr['X'][train_ind], 'PU_label':data_PU_tr['PU_label'][train_ind], 'PN_label':data_PU_tr['PN_label'][train_ind]}

    #scatter plot postive and unlabeled samples
    P_label = data_PU_tr['PU_label'] == 1
    U_N_label = data_PU_tr['PN_label'] == -1
    U_P_label = data_PU_tr['PN_label'] == 1
    plt.scatter(data_PU_tr['X'][U_P_label,0], data_PU_tr['X'][U_P_label,1], alpha=0.5, marker='+', color='gray', s=70,
                label='Unlabeled positive')
    plt.scatter(data_PU_tr['X'][U_N_label,0], data_PU_tr['X'][U_N_label,1], alpha=0.5, marker='_', color='gray', s=70,
                label='Unlabeled negative')
    plt.scatter(data_PU_tr['X'][P_label,0], data_PU_tr['X'][P_label,1], alpha=1, marker='D', color='blue', s=70,
               label='Labeled positive')

    #Contour plot
    Z = (np.array(score_list_wmmd) -2*pi_plus).reshape(d1,d2)
    CS = plt.contour(X, Y, -Z, colors='dodgerblue', levels=[0], linewidths=2)

    #For labeling
    fmt_w = {}
    strs_w = ['WMMD']
    for l, s in zip(CS.levels, strs_w):
        fmt_w[l] = s
    plt.clabel(CS, inline=1, fmt=fmt_w, fontsize=12)

    #Background
    plt.xlim(xmin,xmax)
    plt.ylim(ymin,ymax)
    plt.yticks(fontsize=12)
    plt.xticks(fontsize=12)
    plt.legend(loc='upper right', borderaxespad=0., fontsize=10)

    plt.savefig(outfile, dpi=200, bbox_inches='tight', pad_inches=0)
    print('{} saved'.format(outfile))




def illustrate_decision_boundary_circles(pi_plus, n_p, n_u, device, gamma_list, batch_size, seed=None,
                                         outfile='contour_circles.pdf'):

    print('Plotting the illustration of WMMD decision boundary')
    print('##The training dataset:two_circles')
    print('##class-prior:{}'.format(pi_plus))
    print('##Positive sample size:{}'.format(n_p))
    print('##Unlabeled sample size:{}'.format(n_u))

    #Set seed for reproduciblility
    if seed != None: 
        np.random.seed(seed)

    #generate dataset for training and validation
    data_PU_tr = generate_circles_PU_tr_data(pi_plus, int(1.25*n_p), int(1.25*n_u))

    #For grid range
    xmin = np.floor(np.min(data_PU_tr['X'][:,0]))
    xmax = np.ceil(np.max(data_PU_tr['X'][:,0]))
    ymin = np.floor(np.min(data_PU_tr['X'][:,1]))
    ymax = np.ceil(np.max(data_PU_tr['X'][:,1]))

    #generate delta-grid points
    delta = 0.5
    x = np.arange(xmin, xmax, delta)
    y = np.arange(ymin, ymax, delta)
    X, Y = np.meshgrid(x, y)
    d1, d2 = X.shape
    Xre = X.reshape(-1,1)
    Yre = Y.reshape(-1,1)
    te = np.hstack((Xre,Yre)).astype('float32')

    #Training part
    hyper_wmmd_optimal = [0, 0, 10]  # threshold, gamma, val
    for gam in gamma_list:
        model_wmmd = WMMD(data_dict = data_PU_tr,
                                    device=device, pi_plus=pi_plus,
                                    dict_parameter={'gamma':gam})
        model_wmmd.calculate_val_risk()
        if hyper_wmmd_optimal[2] > model_wmmd.val_loss:
            hyper_wmmd_optimal[0] = model_wmmd.threshold
            hyper_wmmd_optimal[1] = gam
            hyper_wmmd_optimal[2] = model_wmmd.val_loss

    model_wmmd = WMMD(data_dict=data_PU_tr,
                      device=device, pi_plus=pi_plus,
                      dict_parameter={'gamma': hyper_wmmd_optimal[1]})

    #Evaluating the function values
    generator_te = DataLoader(te, batch_size=batch_size, shuffle=False)
    score_list_wmmd = []

    for te_batch in generator_te:
        te_batch = (te_batch).to(device)
        score_wmmd = model_wmmd.score(te_batch).data.cpu().numpy().ravel()
        score_list_wmmd.append(score_wmmd)


    '''
    Plot
    '''
    plt.figure(figsize=(5,5))

    #True function plot
    generated_true = make_circles(2000, factor=0.5)
    P_label = generated_true[1] == 1
    N_label = generated_true[1] == 0
    plt.scatter(generated_true[0][P_label,0]*4, generated_true[0][P_label,1]*4, alpha=0.2, marker='o', color='blue', s=2)
    plt.scatter(generated_true[0][N_label,0]*4, generated_true[0][N_label,1]*4, alpha=0.2, marker='o', color='red', s=2) 

    #scatter plot with train datasets only
    training_data = list(model_wmmd.generator_pu_tr)[0].numpy()
    #print('n_p:{}, n_u:{}'.format(model_wmmd.n_tr_p, model_wmmd.n_tr_u))
    train_ind = [np.where(data_PU_tr['X'] == training_data[i,0:2])[0][0] for i in range(training_data.shape[0])]
    data_PU_tr = {'X':data_PU_tr['X'][train_ind], 'PU_label':data_PU_tr['PU_label'][train_ind], 'PN_label':data_PU_tr['PN_label'][train_ind]}

    #scatter plot postive and unlabeled samples
    P_label = data_PU_tr['PU_label'] == 1
    U_N_label = data_PU_tr['PN_label'] == -1
    U_P_label = data_PU_tr['PN_label'] == 1
    plt.scatter(data_PU_tr['X'][U_P_label,0], data_PU_tr['X'][U_P_label,1], alpha=0.5, marker='+', color='gray', s=70,
                label='Unlabeled positive')
    plt.scatter(data_PU_tr['X'][U_N_label,0], data_PU_tr['X'][U_N_label,1], alpha=0.5, marker='_', color='gray', s=70,
                label='Unlabeled negative')
    plt.scatter(data_PU_tr['X'][P_label,0], data_PU_tr['X'][P_label,1], alpha=1, marker='D', color='blue', s=70,
               label='Labeled positive')

    #Contour plot
    Z = (np.array(score_list_wmmd) -2*pi_plus).reshape(d1,d2)
    CS = plt.contour(X, Y, -Z, colors='dodgerblue', levels=[0], linewidths=2)

    #For labeling
    fmt_w = {}
    strs_w = ['WMMD']
    for l, s in zip(CS.levels, strs_w):
        fmt_w[l] = s
    plt.clabel(CS, inline=1, fmt=fmt_w, fontsize=12)
    
    #Background
    plt.xlim(-7,7)
    plt.ylim(-7,7)
    plt.yticks(fontsize=12)
    plt.xticks(fontsize=12)
    plt.legend(loc='upper right', borderaxespad=0., fontsize=10)

    plt.savefig(outfile, dpi=200, bbox_inches='tight', pad_inches=0)
    print('{} saved'.format(outfile))



def illustrate_decision_boundary_normal(pi_plus, n_p, n_u, device, gamma_list, batch_size, seed=None,
                                         outfile='contour_normal.pdf'):

    print('Plotting the illustration of WMMD decision boundary')
    print('##The training dataset:two_normal')
    print('##class-prior:{}'.format(pi_plus))
    print('##Positive sample size:{}'.format(n_p))
    print('##Unlabeled sample size:{}'.format(n_u))

    #Set seed for reproduciblility
    if seed != None: 
        np.random.seed(seed)

    #Parameters
    mu_p = np.array([1,1])/np.sqrt(2)
    mu_n = np.array([-1,-1])/np.sqrt(2)
    cov_p = np.eye(2)
    cov_n = np.eye(2)


    #generate dataset for training and validation
    data_PU_tr = generate_normal_PU_tr_data(pi_plus, int(1.25*n_p), int(1.25*n_u))

    #For grid range
    xmin = np.floor(np.min(data_PU_tr['X'][:,0]))
    xmax = np.ceil(np.max(data_PU_tr['X'][:,0]))
    ymin = np.floor(np.min(data_PU_tr['X'][:,1]))
    ymax = np.ceil(np.max(data_PU_tr['X'][:,1]))
    xmin, xmax = -4, 3.5
    ymin, ymax = -3.5, 4

    #generate delta-grid points
    delta = 0.5
    x = np.arange(xmin, xmax, delta)
    y = np.arange(ymin, ymax, delta)
    X, Y = np.meshgrid(x, y)
    d1, d2 = X.shape
    Xre = X.reshape(-1,1)
    Yre = Y.reshape(-1,1)
    te = np.hstack((Xre,Yre)).astype('float32')

    #Training part
    hyper_wmmd_optimal = [0,0,10] # threshold, gamma, val            
    for gam in gamma_list:
        model_wmmd = WMMD(data_dict = data_PU_tr,
                                    device=device, pi_plus=pi_plus,
                                    dict_parameter={'gamma':gam})
        model_wmmd.calculate_val_risk()
        if hyper_wmmd_optimal[2] > model_wmmd.val_loss:
            hyper_wmmd_optimal[0] = model_wmmd.threshold
            hyper_wmmd_optimal[1] = gam
            hyper_wmmd_optimal[2] = model_wmmd.val_loss

    model_wmmd = WMMD(data_dict = data_PU_tr,
                      device=device, pi_plus=pi_plus,
                      dict_parameter={'gamma':hyper_wmmd_optimal[1]})

    #Evaluating the function values
    generator_te = DataLoader(te, batch_size=batch_size, shuffle=False)
    score_list_wmmd = []

    for te_batch in generator_te:
        te_batch = (te_batch).to(device)
        score_wmmd = model_wmmd.score(te_batch).data.cpu().numpy().ravel()
        score_list_wmmd.append(score_wmmd)


    '''
    Plot
    '''
    plt.figure(figsize=(5,5))

    #scatter plot with train datasets only
    training_data = list(model_wmmd.generator_pu_tr)[0].numpy()
    #print('n_p:{}, n_u:{}'.format(model_wmmd.n_tr_p, model_wmmd.n_tr_u))
    train_ind = [np.where(data_PU_tr['X'] == training_data[i,0:2])[0][0] for i in range(training_data.shape[0])]
    data_PU_tr = {'X':data_PU_tr['X'][train_ind], 'PU_label':data_PU_tr['PU_label'][train_ind], 'PN_label':data_PU_tr['PN_label'][train_ind]}

    #scatter plot postive and unlabeled samples
    P_label = data_PU_tr['PU_label'] == 1
    U_N_label = data_PU_tr['PN_label'] == -1
    U_P_label = data_PU_tr['PN_label'] == 1
    plt.scatter(data_PU_tr['X'][U_P_label,0], data_PU_tr['X'][U_P_label,1], alpha=0.5, marker='+', color='gray', s=70,
                label='Unlabeled positive')
    plt.scatter(data_PU_tr['X'][U_N_label,0], data_PU_tr['X'][U_N_label,1], alpha=0.5, marker='_', color='gray', s=70,
                label='Unlabeled negative')
    plt.scatter(data_PU_tr['X'][P_label,0], data_PU_tr['X'][P_label,1], alpha=1, marker='D', color='blue', s=70,
               label='Labeled positive')

    #Contour plot
    Z = (np.array(score_list_wmmd) -2*pi_plus).reshape(d1,d2)    
    CS = plt.contour(X, Y, -Z, colors='dodgerblue',levels=[0], linewidths=2)

    #Contour plot for Bayes rule
    delta_p = np.einsum('ij, ij->i', (te-mu_p).dot(np.linalg.inv(cov_p)), (te-mu_p))/2
    delta_n = np.einsum('ij, ij->i', (te-mu_n).dot(np.linalg.inv(cov_n)), (te-mu_n))/2
    Z_bayes = delta_n - delta_p + np.log(pi_plus/(1-pi_plus)) + np.log(np.linalg.det(cov_n)/np.linalg.det(cov_p))/2
    Z_bayes = Z_bayes.reshape(d1,d2)
    CS_bayes = plt.contour(X, Y, Z_bayes, linestyles='dashed', colors='black', levels=[0])

    #For labeling
    fmt_w = {}
    fmt_b = {}
    strs_w = ['WMMD']
    for l, s in zip(CS.levels, strs_w):
        fmt_w[l] = s
    strs_b = ['Bayes']
    for l, s in zip(CS_bayes.levels, strs_b):
        fmt_b[l] = s

    #Plotting
    plt.clabel(CS, inline=1, fmt=fmt_w, fontsize=12, manual=[(1,3)])
    plt.clabel(CS_bayes, inline=1, fmt=fmt_b, fontsize=12, manual=[(1,-3)])
    plt.xlim(xmin,xmax)
    plt.ylim(ymin,ymax)
    plt.yticks(fontsize=12)
    plt.xticks(fontsize=12)
    plt.legend(loc='upper left', borderaxespad=0., fontsize=10)
    plt.savefig(outfile, dpi=200, bbox_inches='tight', pad_inches=0)
    print('{} saved'.format(outfile))

