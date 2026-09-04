from func import log_sum_exp
from PU_DAN.model import ClassifierNetworkD, ClassifierNetworkPhi
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
import math
import matplotlib.pyplot as plt
from cifar_dataset import *
import argparse
import torch
import numpy as np


def train(config):
    is_cuda = torch.cuda.is_available()
    if is_cuda:
        torch.cuda.set_device(config.gpu_id)

    p_loader, x_loader, test_loader = get_cifar_data(batch_size=config.batch_size, num_labeled=config.num_labeled,
                                                     positive_label_list=config.positive_label_list)

    print('----------------------cifar------------------------')
    print('number of train data: ', len(x_loader.dataset))
    print('number of labeled train data: ', len(p_loader.dataset))
    print('number of test data: ', len(test_loader.dataset), '\n')

    iter_num = config.iter_num
    epsilon = config.epsilon
    LR_phi = config.learning_rate_phi
    LR_d = config.learning_rate_d

    model_phi = ClassifierNetworkPhi()
    model_d = ClassifierNetworkD()
    if is_cuda:
        model_phi = model_phi.cuda()
        model_d = model_d.cuda()
    opt_d = torch.optim.Adam(model_d.parameters(), lr=LR_d, betas=(0.5, 0.999))
    opt_phi = torch.optim.Adam(model_phi.parameters(), lr=LR_phi, betas=(0.5, 0.99))

    phi_loss_list = []
    d_loss_list = []

    for epoch in range(iter_num):

        if epoch % 30 == 29:
            LR_d /= 2
            LR_phi /= 2
            opt_d = torch.optim.Adam(model_d.parameters(), lr=LR_d, betas=(0.5, 0.999))
            opt_phi = torch.optim.Adam(model_phi.parameters(), lr=LR_phi, betas=(0.5, 0.99))

        p_phi_ = []
        x_phi_ = []
        phi_batch_loss_list = []
        d_batch_loss_list = []
        kld_batch_list = []

        model_d.train()
        model_phi.train()

        for idx, (X_, _) in enumerate(x_loader):
            P_, _ = list(p_loader)[idx % len(list(p_loader))]

            if is_cuda:
                P_, X_ = P_.cuda(), X_.cuda()

            opt_d.zero_grad()
            W_1 = model_phi(X_)[:, 1].reshape(-1, 1)
            W_P = model_phi(P_)[:, 1]
            W = torch.exp(W_1) / torch.exp(W_1).mean()
            W = W.reshape(-1, 1)
            log_P = model_d(P_)[:, 0]
            log_P = log_P.reshape(-1, 1)
            log_X = model_d(X_)[:, 1]
            log_X = log_X.reshape(-1, 1)
            d_loss = -torch.mean(log_P) - torch.mean(W * log_X)
            d_loss.backward(retain_graph=True)
            opt_d.step()

            opt_phi.zero_grad()
            log_P = model_d(P_)[:, 0]
            log_P = log_P.reshape(-1, 1)
            log_X = model_d(X_)[:, 1]
            log_X = log_X.reshape(-1, 1)
            log_phi1 = log_sum_exp(W_1) - math.log(len(X_))
            phi_loss = torch.abs(torch.mean(log_P) + torch.mean(W * log_X) + 2 * math.log(2)) * (
                    1 - torch.mean(W_P)) / (torch.mean(W_P) - log_phi1 + epsilon)
            kld_ = torch.mean(log_P) + torch.mean(W * log_X) + 2 * math.log(2)
            d_batch_loss_list.append(d_loss.cpu().detach())
            phi_batch_loss_list.append(phi_loss.cpu().detach())
            kld_batch_list.extend([kld_.cpu().data.numpy()])
            phi_loss.backward()
            opt_phi.step()

            p_phi_.append(torch.exp(model_phi(P_)).cpu().data.numpy()[:, 1].mean())
            x_phi_.append(torch.exp(model_phi(X_)).cpu().data.numpy()[:, 1].mean())

        d_loss = np.array(d_batch_loss_list).mean()
        phi_loss = np.array(phi_batch_loss_list).mean()
        d_loss_list.append(d_loss)
        phi_loss_list.append(phi_loss)

        if epoch % 1 == 0:

            print('Train Epoch: {}\td_Loss: {:.6f} phi_loss: {:.6f}'.format(
                epoch, d_loss, phi_loss))
            p_phi_mean = torch.FloatTensor(p_phi_).mean()
            x_phi_mean = torch.FloatTensor(x_phi_).mean()
            print('Train p data phi_mean: {} \tTrain x data phi_mean: {} \tTrain p-x data phi_mean: {}'.format(
                p_phi_mean, x_phi_mean, p_phi_mean - x_phi_mean))

            model_d.eval()
            model_phi.eval()

            # calculate max phi on the train set
            max_phi = 0
            for idx, (data, _) in enumerate(x_loader):
                if is_cuda:
                    data = data.cuda()
                max_phi = max(max_phi, torch.exp(model_phi(data)).max())

            # calculate accuracy on the train set
            num_correct_train = 0
            for data, target in x_loader:
                if is_cuda:
                    data, target = data.cuda(), target.cuda()
                output = torch.exp(model_phi(data))
                output[:, 1] /= max_phi
                output[:, 0] = 1 - output[:, 1]
                pred = output.cpu().data.max(1, keepdim=True)[1]
                num_correct_train += pred.eq(target.cpu().data.view_as(pred)).cpu().sum()
            print('Train set: Accuracy:{}'.format(num_correct_train.numpy() / (len(x_loader.dataset))))

            # calculate accuracy on the test set
            for idx, (data, target) in enumerate(test_loader):
                if is_cuda:
                    data, target = data.cuda(), target.cuda()
                output = torch.exp(model_phi(data))
                output[:, 1] /= max_phi
                output[:, 0] = 1 - output[:, 1]
                if idx == 0:
                    score = output[:, 1].cpu().data
                    pred = output.cpu().data.max(1, keepdim=False)[1]
                    target_ = target.cpu().data
                else:
                    score = torch.cat((score, output[:, 1].cpu().data))
                    pred = torch.cat((pred, output.cpu().data.max(1, keepdim=False)[1]))
                    target_ = torch.cat((target_, target.cpu().data))
            conf_mat = confusion_matrix(target_, pred)
            print('Test set:')
            print('Accuracy:', (conf_mat[0, 0] + conf_mat[1, 1]) / conf_mat.sum())
            print('Confusion matrix:\n', conf_mat)
            print('AUC:', roc_auc_score(target_, score), '\n')


    plt.switch_backend('agg')
    plt.plot(list(range(len(phi_loss_list))), phi_loss_list)
    plt.text(0.5, 0, 'Loss=%.4f' % phi_loss_list[-1], fontdict={'size': 20, 'color': 'red'})


if __name__ == '__main__':
    gpu_id = 0
    batch_size = 500
    epsilon_list = [1e-4]
    lr_phi_list = [3e-5]
    lr_d_list = [3e-5]
    iter_num = 100
    num_labeled = 3000
    positive_label_list = [0, 1, 8, 9]
    for epsilon in epsilon_list:
        for lr_phi in lr_phi_list:
            for lr_d in lr_d_list:
                parser = argparse.ArgumentParser()
                parser.add_argument('--gpu_id', type=int, default=gpu_id)
                parser.add_argument('--batch_size', type=int, default=batch_size)
                parser.add_argument('--epsilon', type=float, default=epsilon)
                parser.add_argument('--num_labeled', type=int, default=num_labeled)
                parser.add_argument('--positive_label_list', type=list, default=positive_label_list)
                parser.add_argument('--learning_rate_phi', type=float, default=lr_phi)
                parser.add_argument('--learning_rate_d', type=float, default=lr_d)
                parser.add_argument('--iter_num', type=int, default=iter_num)
                args = parser.parse_known_args()
                print(args[0], '\n\n')
                train(args[0])