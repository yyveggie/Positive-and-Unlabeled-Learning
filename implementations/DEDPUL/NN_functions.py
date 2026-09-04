import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from random import sample

from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam


def get_discriminator(inp_dim, out_dim=1, hid_dim=32, n_hid_layers=1, bayes=False, bn=True):
    """
    Feed-forward Neural Network constructor
    :param inp_dim: number of input dimensions
    :param out_dim: number of output dimensions; 1 for binary classification
    :param hid_dim: number of hidden dimensions
    :param n_hid_layers: number of hidden layers

    :return: specified neural network
    """

    class Net(nn.Module):
        def __init__(self, inp_dim, out_dim=1, hid_dim=32, n_hid_layers=1, bayes=False, bn=True):
            super(Net, self).__init__()
            self.bayes = bayes
            self.bn = bn
            self.n_hid_layers = n_hid_layers

            self.inp = nn.Linear(inp_dim, hid_dim, bias=not bn)
            if bn:
                self.inp_bn = nn.BatchNorm1d(hid_dim, momentum=0.1)
            if self.n_hid_layers > 0:
                self.hid = nn.Sequential()
                for i in range(n_hid_layers):
                    self.hid.add_module(str(i), nn.Linear(hid_dim, hid_dim, bias=not bn))
                    if bn:
                        self.hid.add_module('bn' + str(i), nn.BatchNorm1d(hid_dim, momentum=0.1))
                    self.hid.add_module('a' + str(i), nn.ReLU())

            if self.bayes:
                self.out_mean = nn.Linear(hid_dim, out_dim)
                self.out_logvar = nn.Linear(hid_dim, out_dim)
            else:
                self.out = nn.Linear(hid_dim, out_dim)

        def forward(self, x, return_params=False, sample_noise=False):
            if self.bn:
                x = F.relu(self.inp_bn(self.inp(x)))
            else:
                x = F.relu(self.inp(x))
            if self.n_hid_layers > 0:
                x = self.hid(x)

            if self.bayes:
                mean, logvar = self.out_mean(x), self.out_logvar(x)
                var = torch.exp(logvar * .5)
                if sample_noise:
                    x = mean + var * torch.randn_like(var)
                else:
                    x = mean
            else:
                mean = self.out(x)
                var = torch.zeros_like(mean) + 1e-3
                x = mean
            p = F.sigmoid(x)

            if return_params:
                return p, mean, var
            else:
                return p
    return Net(inp_dim, out_dim, hid_dim, n_hid_layers, bayes, bn)


def all_convolution(inp_dim=(32, 32, 3), out_dim=1, hid_dim_full=128, bayes=False, bn=True):

    class Net(nn.Module):
        def __init__(self, inp_dim=(32, 32, 3), out_dim=1, hid_dim_full=128, bayes=False, bn=True):
            super(Net, self).__init__()
            self.bayes = bayes
            self.bn = bn

            self.conv1 = nn.Conv2d(3, 16, 5, padding=2)
            self.conv2 = nn.Conv2d(16, 16, 3, padding=1, stride=2)
            self.conv3 = nn.Conv2d(16, 32, 5, padding=2)
            self.conv4 = nn.Conv2d(32, 32, 3, padding=1, stride=2)
            self.conv5 = nn.Conv2d(32, 32, 1)
            self.conv6 = nn.Conv2d(32, 4, 1)
            if bn:
                self.bn1 = nn.BatchNorm2d(16)
                self.bn2 = nn.BatchNorm2d(16)
                self.bn3 = nn.BatchNorm2d(32)
                self.bn4 = nn.BatchNorm2d(32)
                self.bn5 = nn.BatchNorm2d(32)
                self.bn6 = nn.BatchNorm2d(4)

            self.conv_to_fc = 8*8*4
            self.fc1 = nn.Linear(self.conv_to_fc, hid_dim_full)
            self.fc2 = nn.Linear(hid_dim_full, int(hid_dim_full // 2))
            if self.bayes:
                self.out_mean = nn.Linear(int(hid_dim_full // 2), out_dim)
                self.out_logvar = nn.Linear(int(hid_dim_full // 2), out_dim)
            else:
                self.out = nn.Linear(int(hid_dim_full // 2), out_dim)

        def forward(self, x, return_params=False, sample_noise=False):
            if self.bn:
                x = F.relu(self.bn1(self.conv1(x)))
                x = F.relu(self.bn2(self.conv2(x)))
                x = F.relu(self.bn3(self.conv3(x)))
                x = F.relu(self.bn4(self.conv4(x)))
                x = F.relu(self.bn5(self.conv5(x)))
                x = F.relu(self.bn6(self.conv6(x)))
            else:
                x = F.relu(self.conv1(x))
                x = F.relu(self.conv2(x))
                x = F.relu(self.conv3(x))
                x = F.relu(self.conv4(x))
                x = F.relu(self.conv5(x))
                x = F.relu(self.conv6(x))

            x = x.view(-1, self.conv_to_fc)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))

            if self.bayes:
                mean, logvar = self.out_mean(x), self.out_logvar(x)
                var = torch.exp(logvar * .5)
                if sample_noise:
                    x = mean + var * torch.randn_like(var)
                else:
                    x = mean
            else:
                mean = self.out(x)
                var = torch.zeros_like(mean) + 1e-3
                x = mean
            p = F.sigmoid(x)

            if return_params:
                return p, mean, var
            else:
                return p

    return Net(inp_dim, out_dim, hid_dim_full, bayes=bayes, bn=bn)


def d_loss_standard(batch_mix, batch_pos, discriminator, loss_function=None):
    n_mix = batch_mix.shape[0]
    preds = discriminator(torch.cat([batch_mix, batch_pos]))
    d_mix, d_pos = preds[:n_mix], preds[n_mix:]
    if (loss_function is None) or (loss_function == 'log'):
        loss_function = lambda x: torch.log(x + 10 ** -5)  # log loss
    elif loss_function == 'sigmoid':
        loss_function = lambda x: x  # sigmoid loss
    elif loss_function == 'brier':
        loss_function = lambda x: x ** 2  # brier loss
    return -(torch.mean(loss_function(1 - d_pos)) + torch.mean(loss_function(d_mix))) / 2


def KL_normal(m, s):
    return (torch.log(1 / (s + 1e-6)) + (m ** 2 + s ** 2 - 1) * .5).mean()


def d_loss_bayes(batch_mix, batch_pos, discriminator, loss_function=None, w=0.1):
    n_mix = batch_mix.shape[0]
    preds, means, std = discriminator(torch.cat([batch_mix, batch_pos]), return_params=True, sample_noise=True)
    d_mix, d_pos, mean_mix, mean_pos, std_mix, std_pos = preds[:n_mix], preds[n_mix:], means[:n_mix], means[n_mix:], \
                                                         std[:n_mix], std[n_mix:]
    if (loss_function is None) or (loss_function == 'log'):
        loss_function = lambda x: torch.log(x + 10 ** -5)  # log loss
    elif loss_function == 'sigmoid':
        loss_function = lambda x: x  # sigmoid loss
    elif loss_function == 'brier':
        loss_function = lambda x: x ** 2  # brier loss
    loss = -(torch.mean(loss_function(1 - d_pos)) + torch.mean(loss_function(d_mix))) / 2
    loss += (KL_normal(mean_mix, std_mix) + KL_normal(mean_pos, std_pos)) / 2 * w
    return loss


def d_loss_nnRE(batch_mix, batch_pos, discriminator, alpha, beta=0., gamma=1., loss_function=None):
    n_mix = batch_mix.shape[0]
    preds = discriminator(torch.cat([batch_mix, batch_pos]))
    d_mix, d_pos = preds[:n_mix], preds[n_mix:]
    if (loss_function is None) or (loss_function == 'brier'):
        loss_function = lambda x: (1 - x) ** 2  # brier loss
    elif loss_function == 'sigmoid':
        loss_function = lambda x: 1 - x  # sigmoid loss
    elif loss_function in {'log', 'logistic'}:
        loss_function = lambda x: torch.log(1 - x + 10 ** -5)  # log loss
    pos_part = (1 - alpha) * torch.mean(loss_function(1 - d_pos))
    nn_part = torch.mean(loss_function(d_mix)) - (1 - alpha) * torch.mean(loss_function(d_pos))

    if nn_part.item() >= - beta:
        return pos_part + nn_part
    else:
        return -nn_part * gamma


def train_NN(mix_data, pos_data, discriminator, d_optimizer, mix_data_test=None, pos_data_test=None,
             n_epochs=200, batch_size=64, n_batches=None, n_early_stop=5,
             d_scheduler=None, training_mode='standard', disp=False, loss_function=None, nnre_alpha=None,
             metric=None, stop_by_metric=False, bayes=False, bayes_weight=1e-5, beta=0, gamma=1):
    """
    Train discriminator to classify mix_data from pos_data.
    """
    d_losses_train = []
    d_losses_test = []
    d_metrics_test = []
    if n_batches is None:
        n_batches = min(int(mix_data.shape[0] / batch_size), int(pos_data.shape[0] / batch_size))
        batch_size_mix = batch_size_pos = batch_size
    else:
        batch_size_mix, batch_size_pos = int(mix_data.shape[0] / n_batches), int(pos_data.shape[0] / n_batches)
    if mix_data_test is not None:
        data_test = np.concatenate((pos_data_test, mix_data_test))
        target_test = np.concatenate((np.zeros((pos_data_test.shape[0],)), np.ones((mix_data_test.shape[0],))))

    for epoch in range(n_epochs):

        discriminator.train()

        d_losses_cur = []
        if d_scheduler is not None:
            d_scheduler.step()

        for i in range(n_batches):

            batch_mix = np.array(sample(list(mix_data), batch_size_mix))
            batch_pos = np.array(sample(list(pos_data), batch_size_pos))

            batch_mix = torch.as_tensor(batch_mix, dtype=torch.float32)
            batch_pos = torch.as_tensor(batch_pos, dtype=torch.float32)

            # Optimize D
            d_optimizer.zero_grad()

            if training_mode == 'standard':
                if bayes:
                    loss = d_loss_bayes(batch_mix, batch_pos, discriminator, loss_function, bayes_weight)
                else:
                    loss = d_loss_standard(batch_mix, batch_pos, discriminator, loss_function)

            else:
                loss = d_loss_nnRE(batch_mix, batch_pos, discriminator, nnre_alpha, beta=beta, gamma=gamma,
                                   loss_function=loss_function)

            loss.backward()
            d_optimizer.step()
            d_losses_cur.append(loss.cpu().item())

        d_losses_train.append(round(np.mean(d_losses_cur).item(), 5))

        if mix_data_test is not None and pos_data_test is not None:

            discriminator.eval()

            if training_mode == 'standard':
                if bayes:
                    d_losses_test.append(round(d_loss_bayes(torch.as_tensor(mix_data_test, dtype=torch.float32),
                                                            torch.as_tensor(pos_data_test, dtype=torch.float32),
                                                            discriminator, w=bayes_weight).item(), 5))
                else:
                    d_losses_test.append(round(d_loss_standard(torch.as_tensor(mix_data_test, dtype=torch.float32),
                                                               torch.as_tensor(pos_data_test, dtype=torch.float32),
                                                               discriminator).item(), 5))
            elif training_mode == 'nnre':
                d_losses_test.append(round(d_loss_nnRE(torch.as_tensor(mix_data_test, dtype=torch.float32),
                                                       torch.as_tensor(pos_data_test, dtype=torch.float32),
                                                       discriminator, nnre_alpha).item(), 5))
            if metric is not None:
                d_metrics_test.append(metric(target_test,
                                             discriminator(torch.as_tensor(data_test, dtype=torch.float32)).detach().numpy()))

            if disp:
                if not metric:
                    print('epoch', epoch, ', train_loss=', d_losses_train[-1], ', test_loss=', d_losses_test[-1])
                else:
                    print('epoch', epoch, ', train_loss=', d_losses_train[-1], ', test_loss=', d_losses_test[-1],
                          'test_metric=', d_metrics_test[-1])

            if epoch >= n_early_stop:
                if_stop = True
                for i in range(n_early_stop):
                    if metric is not None and stop_by_metric:
                        if d_metrics_test[-i - 1] < d_metrics_test[-n_early_stop - 1]:
                            if_stop = False
                            break
                    else:
                        if d_losses_test[-i-1] < d_losses_test[-n_early_stop-1]:
                            if_stop = False
                            break
                if if_stop:
                    break
        elif disp:
            print('epoch', epoch, ', train_loss=', d_losses_train[-1])

    discriminator.eval()

    return d_losses_train, d_losses_test


def init_keras_model(n_layers=1, n_hid=32, lr=10**-5):
    clf = Sequential()
    for _ in range(n_layers):
        clf.add(Dense(n_hid, activation='relu'))
    clf.add(Dense(1, activation='sigmoid'))
    clf.compile(optimizer=Adam(lr=lr), loss='binary_crossentropy', metrics=['acc'])
    return clf
