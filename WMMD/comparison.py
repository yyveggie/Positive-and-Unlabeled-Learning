#!/usr/bin/env python
# coding: utf-8

#base modules
from sklearn.metrics import roc_auc_score
import numpy as np
from scipy.stats import norm
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import time
from tqdm import tqdm

#torch modules
import torch
from torch.utils.data import DataLoader

#customized module
from wmmd import WMMD
from data import generate_normal_PU_tr_data, generate_normal_te_data

def calculate_acc_AUC_n_u(pi_plus, n_p, n_u_set, n_te, n_repeat, device, gamma_list, batch_size):
    #Output: dictionary with pi_plus_set, accuarcy, AUC, n_u, n_p
    #The shape of accuracy and AUC is [n_repeat, n_u_set]
    print('##Number of positive samples: {}'.format(n_p))
    print('##Class-prior: {}'.format(pi_plus))
    print('##Number of unlabeled samples: np.arange({},{},{})'.format(n_u_set[0], n_u_set[-1]+1, n_u_set[1]-n_u_set[0]))
    print('##Calculating accuracy and AUC...')
    #Initialize the accuracy and AUC
    acc_array_wmmd = np.zeros((n_repeat,len(n_u_set)))
    AUC_array_wmmd = np.zeros((n_repeat,len(n_u_set)))

    for ind_n_rep in tqdm(range(n_repeat)):
        for ind_n_u, n_u in enumerate(n_u_set):

            #Generate normal data
            data_pu_tr = generate_normal_PU_tr_data(pi_plus, n_p, n_u)
            data_te = generate_normal_te_data(pi_plus, n_te)
            generator_te = DataLoader(np.hstack((data_te['X'], data_te['PN_label'].reshape(-1,1))), 
                                      batch_size=batch_size, shuffle=False)

            #Initialization
            Y_te_list = []
            acc_cusum_wmmd = 0
            score_list_wmmd, pred_list_wmmd  = [], []

            #Selecting optimal hyper-parameters
            hyper_wmmd_optimal = [0,0,10] # threshold, gamma, val            
            for gam in gamma_list:
                model_wmmd = WMMD(data_dict = data_pu_tr,
                                  device=device, pi_plus=pi_plus,
                                  dict_parameter={'gamma':gam})

                model_wmmd.calculate_val_risk()
                #print('n_u:{}, val_loss:{}, gamma:{}'.format(n_u, model_wmmd.val_loss, gam))
                if hyper_wmmd_optimal[2] > model_wmmd.val_loss:
                    hyper_wmmd_optimal[0] = model_wmmd.threshold
                    hyper_wmmd_optimal[1] = gam
                    hyper_wmmd_optimal[2] = model_wmmd.val_loss

            #Training
            model_wmmd = WMMD(data_dict = data_pu_tr,
                              device=device, pi_plus=pi_plus,
                              dict_parameter={'gamma':hyper_wmmd_optimal[1]})


            #Test
            for data_te_batch in generator_te:
                X_te_batch, Y_te_batch = data_te_batch[:,:-1], data_te_batch[:,-1]
                X_te_batch = (X_te_batch).to(device)

                Y_te = Y_te_batch.data.numpy()
                Y_te_list.append(Y_te)

                score_wmmd = model_wmmd.score(X_te_batch).data.cpu().numpy().ravel()
                pred_wmmd = (2*((score_wmmd < hyper_wmmd_optimal[0])+0.0) -1)
                acc_cusum_wmmd = np.mean(pred_wmmd == Y_te)
                score_list_wmmd.append(score_wmmd)
                pred_list_wmmd.append(pred_wmmd)

                del X_te_batch

            #Calculate accuracy and AUC
            Y_te_list = np.hstack(Y_te_list)
            score_list_wmmd, pred_list_wmmd  = np.hstack(score_list_wmmd), np.hstack(pred_list_wmmd)
            AUC_wmmd = roc_auc_score(Y_te_list, -score_list_wmmd)
            acc_array_wmmd[ind_n_rep,ind_n_u], AUC_array_wmmd[ind_n_rep,ind_n_u] = acc_cusum_wmmd, AUC_wmmd
            
    return {'n_u_set':n_u_set, 'accuracy':acc_array_wmmd, 'AUC':AUC_array_wmmd, 'pi_plus':pi_plus, 'n_p':n_p}



def calculate_acc_AUC_pi_plus(pi_plus_set, n_p, n_u, n_te, n_repeat, device, gamma_list, batch_size):
    #Output: dictionary with pi_plus_set, accuarcy, AUC, n_u, n_p
    #The shape of accuracy and AUC is [n_repeat, pi_plus_set]
    print('##Number of positive samples: {}'.format(n_p))
    print('##Unlabeled sample size: {}'.format(n_u))
    print('##Class-prior: np.arange({},{},{})'.format(pi_plus_set[0], pi_plus_set[-1]+0.01, pi_plus_set[1]-pi_plus_set[0]))
    print('##Calculating accuracy and AUC...')
    #Initialize the accuracy and AUC
    acc_array_wmmd = np.zeros((n_repeat,len(pi_plus_set)))
    AUC_array_wmmd = np.zeros((n_repeat,len(pi_plus_set)))

    for ind_n_rep in tqdm(range(n_repeat)):
        for ind_pi_plus, pi_plus in enumerate(pi_plus_set):

            #Generate normal data
            data_pu_tr = generate_normal_PU_tr_data(pi_plus, n_p, n_u)
            data_te = generate_normal_te_data(pi_plus, n_te)
            generator_te = DataLoader(np.hstack((data_te['X'], data_te['PN_label'].reshape(-1,1))), 
                                      batch_size=batch_size, shuffle=False)

            #Initialization
            Y_te_list = []
            acc_cusum_wmmd = 0
            score_list_wmmd, pred_list_wmmd  = [], []

            #Selecting optimal hyper-parameters
            hyper_wmmd_optimal = [0,0,10] # threshold, gamma, val            
            for gam in gamma_list:
                model_wmmd = WMMD(data_dict = data_pu_tr,
                                  device=device, pi_plus=pi_plus,
                                  dict_parameter={'gamma':gam})

                model_wmmd.calculate_val_risk()
                #print('n_u:{}, val_loss:{}, gamma:{}'.format(n_u, model_wmmd.val_loss, gam))
                if hyper_wmmd_optimal[2] > model_wmmd.val_loss:
                    hyper_wmmd_optimal[0] = model_wmmd.threshold
                    hyper_wmmd_optimal[1] = gam
                    hyper_wmmd_optimal[2] = model_wmmd.val_loss

            #Training
            model_wmmd = WMMD(data_dict = data_pu_tr,
                              device=device, pi_plus=pi_plus,
                              dict_parameter={'gamma':hyper_wmmd_optimal[1]})


            #Test
            for data_te_batch in generator_te:
                X_te_batch, Y_te_batch = data_te_batch[:,:-1], data_te_batch[:,-1]
                X_te_batch = (X_te_batch).to(device)

                Y_te = Y_te_batch.data.numpy()
                Y_te_list.append(Y_te)

                score_wmmd = model_wmmd.score(X_te_batch).data.cpu().numpy().ravel()
                pred_wmmd = (2*((score_wmmd < hyper_wmmd_optimal[0])+0.0) -1)
                acc_cusum_wmmd = np.mean(pred_wmmd == Y_te)
                score_list_wmmd.append(score_wmmd)
                pred_list_wmmd.append(pred_wmmd)

                del X_te_batch

            #Calculate accuracy and AUC
            Y_te_list = np.hstack(Y_te_list)
            score_list_wmmd, pred_list_wmmd  = np.hstack(score_list_wmmd), np.hstack(pred_list_wmmd)
            AUC_wmmd = roc_auc_score(Y_te_list, -score_list_wmmd)
            acc_array_wmmd[ind_n_rep,ind_pi_plus], AUC_array_wmmd[ind_n_rep,ind_pi_plus] = acc_cusum_wmmd, AUC_wmmd

            
    return {'pi_plus_set':pi_plus_set, 'accuracy':acc_array_wmmd, 'AUC':AUC_array_wmmd, 'n_u':n_u, 'n_p':n_p}


def plot_acc_n_u(pi_plus, n_u_set, acc_array_wmmd, outfile='acc_n_u.pdf'):
    # Input: pi_plus, n_u_set, acc_array_wmmd[n_repeat, len(n_u_set)], outfile
    

    #define axis
    plt.figure(figsize=(7,5))
    plt.xlabel(r"$n_u$: Unlabeled sample size", fontsize=15)
    plt.ylabel("Accuracy (%)", fontsize=15)

    #calculate Bayes risk
    mu_p = np.array([1,1])/np.sqrt(2)
    mu_n = np.array([-1,-1])/np.sqrt(2)
    cov_p = np.eye(2)
    cov_n = np.eye(2)
    delta = np.sqrt(np.dot(np.dot(mu_p-mu_n,np.linalg.inv(cov_p)),mu_p-mu_n))
    log_odds = np.log(pi_plus/(1-pi_plus))
    bayes_risk = pi_plus*norm.cdf(-delta/2-log_odds/delta) +(1-pi_plus)*norm.cdf(-delta/2+log_odds/delta)

    #plot 1-Bayes risk
    plt.plot(n_u_set, 100*(1-bayes_risk)*np.ones(len(n_u_set)), '--', color='black', label='1-Bayes risk')

    #plot WMMD accuracy
    n_repeat = acc_array_wmmd.shape[0]
    ymean = 100*(np.mean(acc_array_wmmd, axis=0))
    yerr = 100*np.std(acc_array_wmmd, axis=0)/np.sqrt(n_repeat)
    plt.plot(n_u_set, ymean, marker='o', color='dodgerblue', label='WMMD') 
    plt.fill_between(n_u_set, ymean-yerr, ymean+yerr, alpha=0.2, edgecolor='dodgerblue', facecolor='dodgerblue')
    plt.legend(loc='lower right', borderaxespad=0., fontsize=13)
    plt.savefig(outfile, dpi=200, bbox_inches='tight', pad_inches=0)
    print('The result plot saved at {}'.format(outfile))


def plot_acc_pi_plus(pi_plus_set, acc_array_wmmd, outfile='acc_pi_plus.pdf'):
    # Input: pi_plus, n_u_set, acc_array_wmmd[n_repeat, len(n_u_set)], outfile
    
    #define axis
    plt.figure(figsize=(7,5))
    plt.xlabel(r"$\pi_{+}$: Class-prior", fontsize=15)
    plt.ylabel("Accuracy (%)", fontsize=15)

    #calculate Bayes risk
    mu_p = np.array([1,1])/np.sqrt(2)
    mu_n = np.array([-1,-1])/np.sqrt(2)
    cov_p = np.eye(2)
    cov_n = np.eye(2)
    bayes_risk = np.zeros(len(pi_plus_set))
    for pi_plus_ind, pi_plus in enumerate(pi_plus_set):
        delta = np.sqrt(np.dot(np.dot(mu_p-mu_n,np.linalg.inv(cov_p)),mu_p-mu_n))
        log_odds = np.log(pi_plus/(1-pi_plus))
        bayes_risk[pi_plus_ind] = pi_plus*norm.cdf(-delta/2-log_odds/delta) +(1-pi_plus)*norm.cdf(-delta/2+log_odds/delta)

    #plot 1-Bayes risk
    plt.plot(pi_plus_set, 100*(1-bayes_risk), '--', color='black', label='1-Bayes risk')

    #plot WMMD accuracy
    n_repeat = acc_array_wmmd.shape[0]
    ymean = 100*(np.mean(acc_array_wmmd, axis=0))
    yerr = 100*np.std(acc_array_wmmd, axis=0)/np.sqrt(n_repeat)
    plt.plot(pi_plus_set, ymean, marker='o', color='dodgerblue', label='WMMD') 
    plt.fill_between(pi_plus_set, ymean-yerr, ymean+yerr, alpha=0.2, edgecolor='dodgerblue', facecolor='dodgerblue')

    plt.legend(loc='upper center', borderaxespad=0., fontsize=15)
    plt.savefig(outfile, dpi=200, bbox_inches='tight', pad_inches=0)
    print('The result plot saved at {}'.format(outfile))



def plot_AUC_n_u(pi_plus, n_u_set, AUC_array_wmmd, outfile='AUC_n_u.pdf'):
    # Input: pi_plus, n_u_set, acc_array_wmmd[n_repeat, len(n_u_set)], outfile
    
    #define axis
    plt.figure(figsize=(7,5))
    plt.xlabel(r"$n_u$: Unlabeled sample size", fontsize=15)
    plt.ylabel(r"AUC $\times 100$", fontsize=15)

    #plot WMMD accuracy
    n_repeat = AUC_array_wmmd.shape[0]
    ymean = 100*(np.mean(AUC_array_wmmd, axis=0))
    yerr = 100*np.std(AUC_array_wmmd, axis=0)/np.sqrt(n_repeat)
    plt.plot(n_u_set, ymean, marker='o', color='dodgerblue', label='WMMD') 
    plt.fill_between(n_u_set, ymean-yerr, ymean+yerr, alpha=0.2, edgecolor='dodgerblue', facecolor='dodgerblue')

    plt.legend(loc='lower right', borderaxespad=0.)
    plt.savefig(outfile, dpi=200, bbox_inches='tight', pad_inches=0)
    print('The result plot saved at {}'.format(outfile))



def plot_AUC_pi_plus(pi_plus_set, AUC_array_wmmd, outfile='AUC_pi_plus.pdf'):
    # Input: pi_plus, n_u_set, acc_array_wmmd[n_repeat, len(n_u_set)], outfile
    
    #define axis
    plt.figure(figsize=(7,5))
    plt.xlabel(r"$\pi_{+}$: Class-prior", fontsize=15)
    plt.ylabel(r"AUC $\times 100$", fontsize=15)

    #plot WMMD accuracy
    n_repeat = AUC_array_wmmd.shape[0]
    ymean = 100*(np.mean(AUC_array_wmmd, axis=0))
    yerr = 100*np.std(AUC_array_wmmd, axis=0)/np.sqrt(n_repeat)
    plt.plot(pi_plus_set, ymean, marker='o', color='dodgerblue', label='WMMD') 
    plt.fill_between(pi_plus_set, ymean-yerr, ymean+yerr, alpha=0.2, edgecolor='dodgerblue', facecolor='dodgerblue')

    plt.legend(loc='lower left', borderaxespad=0.)
    plt.savefig(outfile, dpi=200, bbox_inches='tight', pad_inches=0)
    print('The result plot saved at {}'.format(outfile))


