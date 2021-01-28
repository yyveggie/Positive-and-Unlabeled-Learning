#!/usr/bin/env python
# coding: utf-8

import torch
import argparse
import numpy as np

#customized module
from wmmd import WMMD
from comparison import calculate_acc_AUC_n_u, calculate_acc_AUC_pi_plus, plot_acc_n_u, plot_acc_pi_plus, plot_AUC_n_u, plot_AUC_pi_plus
from illustration import illustrate_decision_boundary_circles, illustrate_decision_boundary_normal, illustrate_decision_boundary_moons

def process_args():
    parser = argparse.ArgumentParser(
        description='Figure implementation of Weighted Maximum Mean Descrepency (WMMD) algorithm for PU learning', 
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batchsize', '-b', type=int, default=10000, 
                        help='Mini batch size')
    parser.add_argument('--gpu', '-g', type=int, default=-1, 
                        help='Zero-origin GPU ID (negative value indicates CPU)')
    parser.add_argument('--preset', '-p', type=str, default=None, 
                        choices=['figure1', 'figure2a', 'figure2b', 'figure2c', 'figure2d'], 
                        help="Preset of configuration\n" + 
                        "figure1, figure2a, figure2b, figure2c, figure2d")
    parser.add_argument('--mode', '-m', type=str, default=None,
                       choices=['accuracy_n_u', 'accuracy_pi_plus', 'AUC_n_u', 'AUC_pi_plus', 
                                'illustration_normal', 'illustration_moons', 'illustration_circles'],
                       help='Mode of figures\n' +
                        'accuracy_n_u: The accuracy on various size of unlabeled samples \n' +
                        'accuracy_pi_plus: The accuracy on various class-prior \n' +
                        'AUC_n_u: The AUC on various size of unlabeled samples \n' +
                        'AUC_pi_plus: The AUC on various class-prior \n' +
                        'illustration_normal: The illustration of the decision boundary with normal training data \n' +
                        'illustration_circles: The illustration of the decision boundary with the two circles training data \n'+
                        'illustration_moons: The illustration of the decision boundary with the two moons training data'
                       )
    parser.add_argument('--replication', '-r', type=int, default=10,
                       help='# of replications for comparison plot')
    parser.add_argument('--testsamples', '-t', type=int, default=1000,
                       help='# of test samples for comparison plot')
    parser.add_argument('--positive', '-P', type=int, default=None,
                       help='# of positive samples')
    parser.add_argument('--unlabeled', '-U', nargs='+', type=int, default=None,
                       help='# of unlabeled samples \n'+
                        'If the mode is accuracy_n_u or AUC_n_u, write the number as "start end by" order')
    parser.add_argument('--classprior', '-c', nargs='+', type=float, default=[0.5],
                       help='class-prior for training and test data \n' +
                        'If the mode is accuracy_pi_plus or AUC_pi_plus, write the number as "start end by" order')
    parser.add_argument('--gammalist', '-G', nargs='+', type=float, default=[1, 0.4, 0.2, 0.1, 0.05],
                        help='list of hyperparameter gamma')
    parser.add_argument('--filename', '-f', type=str, default=None,
                       help='the file name of the result figure')
    parser.add_argument('--directory', '-d', type=str, default='./',
                       help='directory to save the result figure')
    parser.add_argument('--seed', '-s', type=int, default=None,
                       help='random seed for reproducible')
    
    args = parser.parse_args()
    if args.preset == 'figure1':
        args.mode = 'illustration_moons'
        args.classprior = [0.5]
        args.gammalist = [1, 0.4, 0.2, 0.1, 0.05]
        if args.filename == None:
            args.filename = 'figure1.pdf'
    elif args.preset == 'figure2a':
        args.mode = 'accuracy_n_u'
        args.positive = 100
        args.unlabeled = [40, 501, 20]
        args.classprior = [0.5]
        args.gammalist = [1, 0.4, 0.2, 0.1, 0.05]
        args.replication = 100
        args.testsamples = 1000
        if args.filename == None:
            args.filename = 'figure2a.pdf'
    elif args.preset == 'figure2b':
        args.mode = 'accuracy_pi_plus'
        args.positive = 100
        args.unlabeled = [400]
        args.classprior = [0.05, 1, 0.05]
        args.gammalist = [1, 0.4, 0.2, 0.1, 0.05]
        args.replication = 100
        args.testsamples = 1000
        if args.filename == None:
            args.filename = 'figure2b.pdf'
    elif args.preset == 'figure2c':
        args.mode = 'AUC_n_u'
        args.positive = 100
        args.unlabeled = [40, 501, 20]
        args.classprior = [0.5]
        args.gammalist = [1, 0.4, 0.2, 0.1, 0.05]
        args.replication = 100
        args.testsamples = 1000
        if args.filename == None:
            args.filename = 'figure2c.pdf'
    elif args.preset == 'figure2d':
        args.mode = 'AUC_pi_plus'
        args.positive = 100
        args.unlabeled = [400]
        args.classprior = [0.05, 1, 0.05]
        args.gammalist = [1, 0.4, 0.2, 0.1, 0.05]
        args.replication = 100
        args.testsamples = 1000
        if args.filename == None:
            args.filename = 'figure2d.pdf'

    #Default filename
    if args.filename == None:
        args.filename == 'result.pdf'   

    #Assertion errors     
    assert(args.batchsize > 0)
    assert(args.positive != None and args.positive > 0), 'The size of positive samples must be one integer'
    assert(args.replication > 0)
    assert(args.testsamples > 0)
    assert(args.mode != None or args.preset != None), 'Mode or preset must be selected'
    assert(args.unlabeled != None and (len(args.unlabeled) == 1 or len(args.unlabeled)==3)), 'The size of unlabeled samples must be one integer or 3 integers'
    assert(len(args.classprior) == 1 or len(args.classprior) == 3), 'The class-prior must be one integer or 3 integers'
    assert((args.mode.find('pi_plus') >= 0 and len(args.classprior) == 3 and len(args.unlabeled)==1) or
           (args.mode.find('n_u') >= 0 and len(args.unlabeled) == 3 and len(args.classprior)==1) or
           (args.mode.find('illustration')>=0 and len(args.unlabeled)==1 and args.classprior[0]==0.5)
          ), 'Mode and variable are not compatible:\n' + 'For pi_plus mode, class-prior must be 3 integers: start end by \n'+ 'For n_u mode, unlabeled sample size must be 3 integers: start end by \n' + 'For illustration mode, class-prior must be 0.5 and unlabeled sample size must be 1 integer'
    assert(len([n_u for n_u in args.unlabeled if n_u<=0]) == 0), 'The size of unlabeled samples must be a positive integer'
    assert(len([pi_plus for pi_plus in args.classprior if pi_plus<=0.0 or pi_plus>1.0]) == 0), 'The class-prior must be between 0.0 and 1.0'
    assert(len([gamma for gamma in args.unlabeled if gamma<=0.0]) == 0), 'Gamma must be a positive float'
    return args

args = process_args()



def main():
    #Common type variables
    if args.gpu >= 0:
        device = torch.device("cuda:"+str(args.gpu) if torch.cuda.is_available() else "cpu")
        print('Computation hardware:GPU')
        print('Device: cuda:'+str(args.gpu))
    else:
        device = 'cpu'
        print('Computation hardware:CPU')
    n_p = args.positive
    gamma_list = args.gammalist
    n_te = args.testsamples
    n_repeat = args.replication
    batch_size = args.batchsize
    outfile = args.directory + args.filename

    #For each mode...
    if args.mode.find('pi_plus') >= 0:
        pi_plus_set = np.arange(args.classprior[0], args.classprior[1], args.classprior[2])
        n_u = args.unlabeled[0]
        acc_AUC_dict = calculate_acc_AUC_pi_plus(pi_plus_set, n_p, n_u, n_te, n_repeat, device, gamma_list, batch_size)
        if args.mode.find('accuracy') >= 0:
            plot_acc_pi_plus(acc_AUC_dict['pi_plus_set'], acc_AUC_dict['accuracy'], outfile)
        else:
            plot_AUC_pi_plus(acc_AUC_dict['pi_plus_set'], acc_AUC_dict['AUC'], outfile)

    elif args.mode.find('n_u') >= 0:
        n_u_set = np.arange(args.unlabeled[0], args.unlabeled[1], args.unlabeled[2])
        pi_plus = args.classprior[0]
        print(pi_plus)
        breakpoint()
        acc_AUC_dict = calculate_acc_AUC_n_u(pi_plus, n_p, n_u_set, n_te, n_repeat, device, gamma_list, batch_size)
        if args.mode.find('accuracy') >= 0:
            plot_acc_n_u(acc_AUC_dict['pi_plus'], acc_AUC_dict['n_u_set'], acc_AUC_dict['accuracy'], outfile)
        else:
            plot_AUC_n_u(acc_AUC_dict['pi_plus'], acc_AUC_dict['n_u_set'], acc_AUC_dict['AUC'], outfile)

    elif args.mode.find('normal') >= 0:
        pi_plus = args.classprior[0]
        n_u = args.unlabeled[0]
        illustrate_decision_boundary_normal(pi_plus, n_p, n_u, device, gamma_list, batch_size, args.seed, outfile)

    elif args.mode.find('circles') >= 0:
        pi_plus = args.classprior[0]
        n_u = args.unlabeled[0]
        illustrate_decision_boundary_circles(pi_plus, n_p, n_u, device, gamma_list, batch_size, args.seed, outfile)

    else:
        pi_plus = args.classprior[0]
        n_u = args.unlabeled[0]
        illustrate_decision_boundary_moons(pi_plus, n_p, n_u, device, gamma_list, batch_size, args.seed, outfile)

if __name__ == '__main__':
    main()