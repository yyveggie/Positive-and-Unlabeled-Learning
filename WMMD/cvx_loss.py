#!/usr/bin/env python3

"""
cvx_losses.py: This code implements the several convex loss based PU learning algorithms
proposed in the paper 'Convex Formulation for Learning from Positive
and Unlabeled Data' by du Plessis et. al. (2015), ICML.
The following functions assume that training set is not big enough to conduct
a full-batch algorithm. (The gram matrix is not big)
"""

import torch
import numpy as np

class CVX_Loss(object):
    def __init__(self, data_array=None,
                device=None, pi_plus=0.5,
                dict_parameter={'lam':0.1,'gamma':0.1}):
        self.device = device

        # data_array assumes the last index is PU indicator
        assert (data_array is not None), 'check data_array'

        # hyperparameter
        self.lam = dict_parameter['lam']
        self.gamma = dict_parameter['gamma']
        self.pi_plus = (pi_plus*torch.ones([1])).to(self.device)[0]

        # Datasets
        X, PU_indicator = data_array[:,:-1], data_array[:,-1]
        X_p, X_u = X[PU_indicator==1], X[PU_indicator==0]        
        self.n_p, self.n_u = X_p.shape[0], X_u.shape[0]

        # Split training and validation set
        ind_p = np.random.choice(self.n_p, size=(self.n_p//5), replace=False)
        ind_u = np.random.choice(self.n_u, size=(self.n_u//5), replace=False)
        X_tr_p, X_val_p = X_p[[z for z in range(self.n_p) if not z in ind_p]], X_p[ind_p]
        X_tr_u, X_val_u = X_u[[z for z in range(self.n_u) if not z in ind_u]], X_u[ind_u]

        self.n_tr_p, self.n_tr_u = X_tr_p.shape[0], X_tr_u.shape[0]
        self.n_val_p, self.n_val_u = X_val_p.shape[0], X_val_u.shape[0]

        self.X_tr = torch.from_numpy(np.vstack([X_tr_p, X_tr_u])).to(self.device)
        self.X_val = torch.from_numpy(np.vstack([X_val_p, X_val_u])).to(self.device)

        self.n_tr = self.n_tr_p + self.n_tr_u
        self.n_val = self.n_val_p + self.n_val_u

        # initialize
        self.alpha = torch.randn(self.n_tr, 1, requires_grad=True, device=self.device)
        self.best_alpha = torch.randn(self.n_tr, 1, requires_grad=True, device=self.device)
        self.intercept = torch.randn(1, 1, requires_grad=True, device=self.device)
        self.best_intercept = torch.randn(1, 1, requires_grad=True, device=self.device)

        self.tr_loss, self.val_loss = 0, 0

    def print_validation_loss(self):
        print('lambda: ', self.lam)
        print('gamma: ', self.gamma)
        print('train loss: ', self.tr_loss)
        print('validation loss: ', self.val_loss)

    def calculate_gram(self):
        ZZT = torch.mm(self.X_tr, self.X_tr.t())
        diag_ZZT = torch.diag(ZZT).unsqueeze(1)
        Z_norm_sqr = diag_ZZT.expand_as(ZZT)
        exponent = Z_norm_sqr - 2 * ZZT + Z_norm_sqr.t()
        gram = torch.exp(-self.gamma*exponent)
        return gram

    def score(self, Z):
        '''
        calculate the binary discriminant function value, often called score
        '''
        Z = Z.to(self.device)
        n_z = Z.size()[0]

        X_tr_sqr = torch.sum(self.X_tr * self.X_tr, dim=1).repeat(n_z, 1).view(n_z, self.n_tr)
        X_val_sqr = torch.sum(Z * Z, dim=1).view(-1,1).repeat(1, self.n_tr).view(n_z, self.n_tr)
        interaction = torch.mm(Z, self.X_tr.t())
        exponent = X_tr_sqr + X_val_sqr - 2*interaction
        
        kernel_matrix = torch.exp(-self.gamma*exponent)
        score_value = torch.mm(kernel_matrix, self.alpha) + self.intercept # n_z by 1

        return score_value

    def predict(self, score):
        predict = score > 0
        return predict


class Logistic_Loss(CVX_Loss):
    """
    This class implements the 'logistic loss' based PU learning algorithm
    proposed in the paper 'Convex Formulation for Learning from Positive
    and Unlabeled Data' by du Plessis et. al. (2015), ICML.
    """

    def __init__(self, n_epochs=10, lr= 0.01, n_patience=10, verbose=1, *args, **kwargs):
        super(Logistic_Loss, self).__init__(*args, **kwargs)

        self.lr = lr
        self.n_epochs = n_epochs
        self.verbose = verbose
        self.n_patience = n_patience
        self.gram = self.calculate_gram()
        self.gram_p, self.gram_u = self.gram[:self.n_tr_p], self.gram[self.n_tr_p:]


    def log_loss(self, Z, data_type='val'):
        if data_type == 'tr':
            score_value_p = torch.mm(self.gram_p,self.alpha) + self.intercept
            score_value_u = torch.mm(self.gram_u,self.alpha) + self.intercept
        elif data_type == 'val':
            score_value = self.score(Z)
            score_value_p, score_value_u = score_value[:self.n_val_p], score_value[self.n_val_p:]
        else:
            assert False, 'Please check data_type'
        
        # calculate square loss
        part_1 = -self.pi_plus*torch.mean(score_value_p)
        z = -score_value_u # logistic loss
        part_2 = torch.mean(torch.log(1 + torch.exp(-z)))
        reg = self.lam*torch.mm(self.alpha.t(), self.alpha)/2
        loss = part_1 + part_2 + reg
        return loss


    def fit(self):
        # gradient descent algorithm
        best_val_loss = 100
        count_patience = 0
        optimizer = torch.optim.SGD([self.alpha, self.intercept], lr=self.lr)
        for _ in range(self.n_epochs):
            
            optimizer.zero_grad()
            tr_loss = self.log_loss(self.X_tr, data_type='tr')
            val_loss = self.log_loss(self.X_val, data_type='val')
            if self.verbose == 1:
                print('tr/val loss: ', tr_loss, '/', val_loss)
            tr_loss.backward()
            optimizer.step()

            if best_val_loss > val_loss:
                best_val_loss = val_loss
                count_patience = 0
                self.best_alpha = self.alpha
                self.best_intercept = self.intercept
            else:
                count_patience += 1

            if count_patience >= self.n_patience:
                # print('Complete training')
                self.alpha = self.best_alpha
                self.intercept = self.best_intercept
                self.tr_loss = tr_loss.data.cpu().numpy().ravel()
                self.val_loss = val_loss.data.cpu().numpy().ravel()
                break

        # print('Complete training')
        self.alpha = self.best_alpha
        self.intercept = self.best_intercept
        self.tr_loss = tr_loss.data.cpu().numpy().ravel()
        self.val_loss = val_loss.data.cpu().numpy().ravel()

class Double_Hinge_Loss(CVX_Loss):
    """
    This class implements the 'double hinge loss' based PU learning algorithm
    proposed in the paper 'Convex Formulation for Learning from Positive
    and Unlabeled Data' by du Plessis et. al. (2015), ICML.
    """

    def __init__(self, n_epochs=10, lr= 0.01, n_patience=10, verbose=1, *args, **kwargs):
        super(Double_Hinge_Loss, self).__init__(*args, **kwargs)

        self.lr = lr
        self.n_epochs = n_epochs
        self.verbose = verbose
        self.n_patience = n_patience
        self.gram = self.calculate_gram()
        self.gram_p, self.gram_u = self.gram[:self.n_tr_p], self.gram[self.n_tr_p:]


    def hinge_loss(self, Z, data_type='val'):
        if data_type == 'tr':
            score_value_p = torch.mm(self.gram_p,self.alpha) + self.intercept
            score_value_u = torch.mm(self.gram_u,self.alpha) + self.intercept
        elif data_type == 'val':
            score_value = self.score(Z)
            score_value_p, score_value_u = score_value[:self.n_val_p], score_value[self.n_val_p:]
        else:
            assert False, 'Please check data_type'
        
        # calculate square loss
        part_1 = -self.pi_plus*torch.mean(score_value_p)
        z = -score_value_u # logistic loss
        part_2 = torch.mean(torch.relu(torch.max(-z, 0.5-0.5*z))) # double hinge loss
        reg = self.lam*torch.mm(self.alpha.t(), self.alpha)/2
        loss = part_1 + part_2 + reg
        return loss    


    def fit(self):
        # gradient descent algorithm
        best_val_loss = 100
        count_patience = 0
        optimizer = torch.optim.SGD([self.alpha, self.intercept], lr=self.lr)
        for _ in range(self.n_epochs):
            
            optimizer.zero_grad()
            tr_loss = self.hinge_loss(self.X_tr, data_type='tr')
            val_loss = self.hinge_loss(self.X_val, data_type='val')
            if self.verbose == 1:
                print('tr/val loss: ', tr_loss, '/', val_loss)
            tr_loss.backward()
            optimizer.step()

            if best_val_loss > val_loss:
                best_val_loss = val_loss
                count_patience = 0
                self.best_alpha = self.alpha
                self.best_intercept = self.intercept
            else:
                count_patience += 1

            if count_patience >= self.n_patience:
                # print('Complete training')
                self.alpha = self.best_alpha
                self.intercept = self.best_intercept
                self.tr_loss = tr_loss.data.cpu().numpy().ravel()
                self.val_loss = val_loss.data.cpu().numpy().ravel()
                break
        
        # print('Complete training')
        self.alpha = self.best_alpha
        self.intercept = self.best_intercept
        self.tr_loss = tr_loss.data.cpu().numpy().ravel()
        self.val_loss = val_loss.data.cpu().numpy().ravel()

