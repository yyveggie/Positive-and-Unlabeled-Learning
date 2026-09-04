import torch
import numpy as np
from torch.utils.data import DataLoader




class WMMD(object):
    def __init__(self, data_dict=None,
                device=None, pi_plus=0.5,
                batch_size=1024, n_folds=5,
                dict_parameter={'gamma': 0.1}):
        self.device = device

        # hyperparameter
        self.gamma = dict_parameter['gamma']
        self.batch_size = batch_size

        self.threshold = 2*pi_plus

        self.pi_plus = (pi_plus*torch.ones([1])).to(self.device)[0]

        # Datasets
        X, PU_indicator = data_dict['X'], data_dict['PU_label']
        X_p, X_u = X[PU_indicator==1], X[PU_indicator==0]        
        n_p, n_u = X_p.shape[0], X_u.shape[0]

        # Split training and validation set
        ind_p = np.random.choice(n_p, size=(n_p//n_folds), replace=False)
        ind_u = np.random.choice(n_u, size=(n_u//n_folds), replace=False)
        X_tr_p, X_val_p = X_p[[z for z in range(n_p) if not z in ind_p]], X_p[ind_p]
        X_tr_u, X_val_u = X_u[[z for z in range(n_u) if not z in ind_u]], X_u[ind_u]

        self.n_tr_p, self.n_tr_u = X_tr_p.shape[0], X_tr_u.shape[0]
        self.n_val_p, self.n_val_u = X_val_p.shape[0], X_val_u.shape[0]

        self.n_tr = self.n_tr_p + self.n_tr_u
        self.n_val = self.n_val_p + self.n_val_u

        X_tr = np.vstack([X_tr_p, X_tr_u])
        pu_tr_label = np.hstack((np.ones(self.n_tr_p),np.zeros(self.n_tr_u)))
        data_pu_tr = np.hstack((X_tr, pu_tr_label[:,None])).astype('float32')

        X_val = np.vstack([X_val_p, X_val_u])
        pu_val_label = np.hstack((np.ones(self.n_val_p),np.zeros(self.n_val_u)))
        data_pu_val = np.hstack((X_val, pu_val_label[:,None])).astype('float32')

        self.generator_pu_tr = DataLoader(data_pu_tr, batch_size=batch_size, shuffle=False)
        self.generator_pu_val = DataLoader(data_pu_val, batch_size=batch_size, shuffle=False)

        self.val_loss = 0

    # For tuning hyper parameters
    def calculate_val_risk(self):
        score_p, score_u = [], []
        for data_val_batch in self.generator_pu_val:
            X_val_batch, ind_postive_batch = data_val_batch[:,:-1], data_val_batch[:,-1]

            X_p_batch, X_u_batch = X_val_batch[ind_postive_batch==1], X_val_batch[ind_postive_batch==0]

            if X_p_batch.shape[0] > 0:
                X_p_batch = (X_p_batch).to(self.device)
                score_p.append(self.score(X_p_batch).data.cpu().numpy().ravel())

            if X_u_batch.shape[0] > 0:
                X_u_batch = (X_u_batch).to(self.device)
                score_u.append(self.score(X_u_batch).data.cpu().numpy().ravel())
        
        score_p, score_u = np.hstack(score_p).ravel(), np.hstack(score_u).ravel()

        score_all = np.hstack([score_p, score_u])

        p = self.pi_plus.data.cpu().numpy()
        Q_1 = np.mean(score_p > 2*p)
        Q_X = np.mean(score_u < 2*p)
        risk = -p + 2*p*Q_1 + Q_X
        self.val_loss = risk
        

    def sum_of_gaussian_kernel(self, Z_te, Z_tr):
        """
        This function calculates the sum of kernel values between Z_te and Z_tr
        """
        if Z_te.size()[0]>0 and Z_tr.size()[0]>0:
            #Check whether the batch is empty.
            X_tr_sqr = torch.sum(Z_tr * Z_tr, dim=1).repeat(Z_te.size()[0], 1).view(Z_te.size()[0],Z_tr.size()[0])
            X_te_sqr = torch.sum(Z_te * Z_te, dim=1).view(-1,1).repeat(1, Z_tr.size()[0]).view(Z_te.size()[0],Z_tr.size()[0])
            interaction = torch.mm(Z_te, Z_tr.t())
            exponent = X_tr_sqr + X_te_sqr - 2*interaction
            
            kernel_matrix = torch.exp(-self.gamma*exponent)
            sum_of_kernel = torch.sum(kernel_matrix, dim=1)

        else:
            #Returns 0 if training batch is empty.
            sum_of_kernel = torch.tensor([0], dtype=Z_tr.dtype, device=Z_tr.device)

        return sum_of_kernel

    def gaussian_kernel_calculator(self, X_te_batch, X_p_batch, X_u_batch):
        """
        This function returns the two sum of kernels.
        """
        numer_part = self.sum_of_gaussian_kernel(X_te_batch, X_u_batch)
        denom_part = self.sum_of_gaussian_kernel(X_te_batch, X_p_batch)

        return numer_part, denom_part

    def score(self, X_te_batch):
        numer_cusum, denom_cusum = 0, 0
        for data_tr_batch in self.generator_pu_tr:
            X_tr_batch, ind_postive_batch = data_tr_batch[:,:-1], data_tr_batch[:,-1]

            X_p_batch, X_u_batch = X_tr_batch[ind_postive_batch==1], X_tr_batch[ind_postive_batch==0]
            X_p_batch, X_u_batch = (X_p_batch).to(self.device), (X_u_batch).to(self.device)     

            numer_part, denom_part = self.gaussian_kernel_calculator(X_te_batch, X_p_batch, X_u_batch)
            del X_u_batch, X_p_batch
            numer_cusum += numer_part; del numer_part
            denom_cusum += denom_part; del denom_part

        numer = numer_cusum/self.n_tr_u 
        denom = denom_cusum/self.n_tr_p + 1e-8

        return numer/denom
        

    def predict(self, score):
        predict = score < self.threshold
        return predict

if __name__ == '__main__':
    WMMD()