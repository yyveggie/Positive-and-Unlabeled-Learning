from itertools import repeat, cycle
import numpy as np
import argparse
import torch as pt
#import torchvision as ptv
import torch
import torch.nn as nn
from torch.autograd import Variable
import losses,ramps
from collections import OrderedDict
import pickle
import os
import random
from MLP import MLP
#from RN_mining import RN_mining
import torch.utils.data
from dataset import load_dataset

#use_cuda = torch.cuda.is_available()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def mixup_data(x, y, alpha=1.0):
    '''Compute the mixup data. Return mixed inputs, mixed target, and lambda'''
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.
    batch_size = x.size()[0]
    index = np.random.permutation(batch_size)
    x, y = x.data.cpu().numpy(), y.data.cpu().numpy()
    mixed_x = torch.Tensor(lam * x + (1 - lam) * x[index, :])
    mixed_y = torch.Tensor(lam * y + (1 - lam) * y[index, :])

    mixed_x = Variable(mixed_x.to(device))
    mixed_y = Variable(mixed_y.to(device))
    return mixed_x, mixed_y, lam


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    # labeled_minibatch_size = max(target.ne(NO_LABEL).sum(), 1e-8).type(torch.cuda.FloatTensor)
    minibatch_size = len(target)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / minibatch_size))
    return res


def mixup_data_sup(x, y, alpha=1.0):
    '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.
    batch_size = x.size()[0]
    index = np.random.permutation(batch_size)
    # x, y = x.numpy(), y.numpy()
    # mixed_x = torch.Tensor(lam * x + (1 - lam) * x[index,:])
    mixed_x = lam * x + (1 - lam) * x[index, :]
    # y_a, y_b = torch.Tensor(y).type(torch.LongTensor), torch.Tensor(y[index]).type(torch.LongTensor)
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def get_current_consistency_weight(final_consistency_weight, epoch, step_in_epoch, total_steps_in_epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    epoch = epoch - args.consistency_rampup_starts
    epoch = epoch + step_in_epoch / total_steps_in_epoch
    return final_consistency_weight * ramps.sigmoid_rampup(epoch,
                                                           args.consistency_rampup_ends - args.consistency_rampup_starts)

def mixup_criterion(y_a, y_b, lam):
    return lambda criterion, pred: lam * criterion(pred, y_a.long()) + (1 - lam) * criterion(pred, y_b.long())


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


parser = argparse.ArgumentParser(description='Interpolation consistency training')

parser.add_argument('--pseudo_label', choices=['single', 'mean_teacher'],
                    help='pseudo label generated from either a single model or mean teacher model')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--ema_decay', default=0.999, type=float, metavar='ALPHA',
                    help='ema variable decay rate (default: 0.999)')
parser.add_argument('--mixup_consistency', default=1.0, type=float,
                    help='consistency coeff for mixup usup loss')
parser.add_argument('--consistency_type', default="mse", type=str, metavar='TYPE',
                    choices=['mse', 'kl'],
                    help='consistency loss type to use')
parser.add_argument('--consistency_rampup_starts', default=1, type=int, metavar='EPOCHS',
                    help='epoch at which consistency loss ramp-up starts')
parser.add_argument('--consistency_rampup_ends', default=1, type=int, metavar='EPOCHS',
                    help='lepoch at which consistency loss ramp-up ends')
parser.add_argument('--mixup_sup_alpha', default=1.0, type=float,
                    help='for supervised loss, the alpha parameter for the beta distribution from where the mixing lambda is drawn')
parser.add_argument('--mixup_usup_alpha', default=2.0, type=float,
                    help='for unsupervised loss, the alpha parameter for the beta distribution from where the mixing lambda is drawn')
parser.add_argument('--mixup_hidden', action='store_true',
                    help='apply mixup in hidden layers')
parser.add_argument('--num_mix_layer', default=3, type=int,
                    help='number of layers on which mixup is applied including input layer')
parser.add_argument('--evaluation_epochs', default=1, type=int,
                    metavar='EPOCHS', help='evaluation frequency in epochs, 0 to turn evaluation off (default: 1)')
parser.add_argument('--print_freq', '-p', default=100, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--batchsize', '-b', type=int, default=30000,
                        help='Mini batch size')
parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='Zero-origin GPU ID (negative value indicates CPU)')
parser.add_argument('--dataset', '-d', default='mnist', type=str, choices=['mnist', 'cifar10'],
                        help='The dataset name')
parser.add_argument('--labeled', '-l', default=100, type=int,
                        help='# of labeled data')
parser.add_argument('--unlabeled', '-u', default=49900, type=int,
                        help='# of unlabeled data')
parser.add_argument('--epoch', '-e', default=100, type=int,
                        help='# of epochs to learn')
parser.add_argument('--beta', '-B', default=0., type=float,
                        help='Beta parameter of nnPU')
parser.add_argument('--gamma', '-G', default=1., type=float,
                        help='Gamma parameter of nnPU')
parser.add_argument('--loss', type=str, default="sigmoid", choices=['logistic', 'sigmoid'],
                        help='The name of a loss function')
parser.add_argument('--model', '-m', default='3lp', choices=['linear', '3lp', 'mlp'],
                        help='The name of a classification model')
parser.add_argument('--stepsize', '-s', default=1e-3, type=float,
                        help='Stepsize of gradient method')
parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')

best_prec1 = 0
global_step = 0
args = parser.parse_args()
use_cuda = torch.cuda.is_available()

'''
# Networks for MNIST
class MLP(pt.nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = pt.nn.Linear(784, 512)
        self.fc2 = pt.nn.Linear(512, 128)
        self.fc3 = pt.nn.Linear(128, 2)

    def forward(self, din):
        din = din.view(-1, 28 * 28)
        dout = pt.nn.functional.relu(self.fc1(din))
        dout = pt.nn.functional.relu(self.fc2(dout))
        return pt.nn.functional.softmax(self.fc3(dout))
'''

def pre_train(trainloader, model, optimizer, epoch):
    import utils
    
    # switch to train mode
    model.train()
    
    meters = utils.AverageMeterSet()
    class_criterion = nn.CrossEntropyLoss().to(device)
    i = -1
    for (input, target) in trainloader:

        #if False:
        if args.mixup_sup_alpha:
            if use_cuda:
                input, target = input.to(device), target.to(device)
            input_var, target_var = Variable(input), Variable(target)

            if args.mixup_hidden:
                ### model
                output_mixed_l, target_a_var, target_b_var, lam = model(input_var, target_var, mixup_hidden=True,
                                                                        mixup_alpha=args.mixup_sup_alpha,
                                                                        layers_mix=args.num_mix_layer)
                lam = lam[0]
            else:
                mixed_input, target_a, target_b, lam = mixup_data_sup(input, target, args.mixup_sup_alpha)
                # if use_cuda:
                #    mixed_input, target_a, target_b  = mixed_input.cuda(), target_a.cuda(), target_b.cuda()
                mixed_input_var, target_a_var, target_b_var = Variable(mixed_input), Variable(target_a), Variable(target_b)
                ### model
                output_mixed_l = model(mixed_input_var)

            loss_func = mixup_criterion(target_a_var, target_b_var, lam)
            class_loss = loss_func(class_criterion, output_mixed_l)

        else:
            input_var = torch.autograd.Variable(input.to(device))

            target_var = torch.autograd.Variable(target.to(device))

        #    print (input_var.shape)
        #    print (type(input_var))
            output = model(input_var.float())

            class_loss = class_criterion(output, target_var.long())

        #print("class_loss",class_loss)
        meters.update('class_loss', class_loss.item())
        loss = class_loss
        #print ("loss",loss)

        ### get ema loss. We use the actual samples(not the mixed up samples ) for calculating EMA loss
        minibatch_size = len(target_var)

        if args.mixup_sup_alpha:
            class_logit = model(input_var)
        else:
            class_logit = output


        meters.update('loss', loss.item())

        prec1, prec5 = accuracy(class_logit.data, target_var.data, topk=(1, 2))

        meters.update('top1', prec1[0], minibatch_size)
        meters.update('error1', 100.0 - prec1[0], minibatch_size)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_class_loss_list.append(meters['class_loss'].avg)
    train_error_list.append(float(meters['error1'].avg))
    train_pre_list.append(meters['top1'].avg)
        # measure elapsed time
        #meters.update('batch_time', time.time() - end)
        #end = time.time()

        #print ("prec1, prec5",prec1, prec5)

def train(trainloader, unlabelledloader, model, ema_model, optimizer, epoch):

    global global_step
    if args.consistency_type == 'mse':
        consistency_criterion = losses.softmax_mse_loss
    elif args.consistency_type == 'kl':
        consistency_criterion = losses.softmax_kl_loss
    else:
        assert False, args.consistency_type

    # switch to train mode
    model.train()
    ema_model.train()
        
        
    import utils
    import time
    meters = utils.AverageMeterSet()
    #class_criterion=nn.MSELoss().cuda()
    class_criterion = nn.CrossEntropyLoss().to(device)
    i = -1

    for (input, target), (u, _) in zip(cycle(trainloader), unlabelledloader):

        #print ("target",target[0:6])
        i = i + 1
        if input.shape[0] != u.shape[0]:
            bt_size = np.minimum(input.shape[0], u.shape[0])
            input = input[0:bt_size]
            target = target[0:bt_size]
            u = u[0:bt_size]


        if args.mixup_sup_alpha:
            if use_cuda:
                input, target, u = input.to(device), target.to(device), u.to(device)
            input_var, target_var, u_var = Variable(input), Variable(target), Variable(u)

            #IF False:
            if args.mixup_hidden:
                ### model
                output_mixed_l, target_a_var, target_b_var, lam = model(input_var, target_var, mixup_hidden=True,
                                                                        mixup_alpha=args.mixup_sup_alpha,
                                                                        layers_mix=args.num_mix_layer)
                lam = lam[0]
            else:
                mixed_input, target_a, target_b, lam = mixup_data_sup(input, target, args.mixup_sup_alpha)
                # if use_cuda:
                #    mixed_input, target_a, target_b  = mixed_input.cuda(), target_a.cuda(), target_b.cuda()
                mixed_input_var, target_a_var, target_b_var = Variable(mixed_input), Variable(target_a), Variable(target_b)
                ### model
                output_mixed_l = model(mixed_input_var.float())

            loss_func = mixup_criterion(target_a_var, target_b_var, lam)
            class_loss = loss_func(class_criterion, output_mixed_l)
            output = output_mixed_l

        else:
            input_var = torch.autograd.Variable(input.to(device))
            with torch.no_grad():
                u_var = torch.autograd.Variable(u.to(device))
            target_var = torch.autograd.Variable(target.to(device))
            ### model
            output = model(input_var.float())

            # sharpening
            #output = output**2 / sum([x**2 for x in output])

            #print ("output",output[0:6])
            #print ("target",target[0:6])
            class_loss = class_criterion(output, target_var.long()) / len(output)

        #print("class_loss",class_loss)
        meters.update('class_loss', class_loss.item())

        ### get ema loss. We use the actual samples(not the mixed up samples ) for calculating EMA loss
        minibatch_size = len(target_var)
        if args.pseudo_label == 'single':
            ema_logit_unlabeled = model(u_var.float())
            ema_logit_labeled = model(input_var.float())
        else:
            ema_logit_unlabeled = ema_model(u_var.float())
            ema_logit_labeled = ema_model(input_var.float())
        if args.mixup_sup_alpha:
            class_logit = model(input_var.float())
        else:
            class_logit = output
        cons_logit = model(u_var.float())
        #print ("cons_logit",cons_logit)

        ema_logit_unlabeled = Variable(ema_logit_unlabeled.detach().data, requires_grad=False)

        # class_loss = class_criterion(class_logit, target_var) / minibatch_size

        ema_class_loss = class_criterion(ema_logit_labeled, target_var.long())  # / minibatch_size
        meters.update('ema_class_loss', ema_class_loss.item())
        #print ("ema_class_loss",ema_class_loss)


        ### get the unsupervised mixup loss###
        if args.mixup_consistency:
            if args.mixup_hidden:
                # output_u = model(u_var)
                output_mixed_u, target_a_var, target_b_var, lam = model(u_var.float(), ema_logit_unlabeled,
                                                                        mixup_hidden=True,
                                                                        mixup_alpha=args.mixup_sup_alpha,
                                                                        layers_mix=args.num_mix_layer)
                # ema_logit_unlabeled
                lam = lam[0]
                mixedup_target = lam * target_a_var + (1 - lam) * target_b_var
            else:
                # output_u = model(u_var)
                mixedup_x, mixedup_target, lam = mixup_data(u_var, ema_logit_unlabeled, args.mixup_usup_alpha)
                # mixedup_x, mixedup_target, lam = mixup_data(u_var, output_u, args.mixup_usup_alpha)
                output_mixed_u = model(mixedup_x.float())
            mixup_consistency_loss = consistency_criterion(output_mixed_u,
                                                           mixedup_target) / minibatch_size  # criterion_u(F.log_softmax(output_mixed_u,1), F.softmax(mixedup_target,1))
            meters.update('mixup_cons_loss', mixup_consistency_loss.item())
            if epoch < args.consistency_rampup_starts:
                mixup_consistency_weight = 0.0
            else:
                mixup_consistency_weight = get_current_consistency_weight(args.mixup_consistency, epoch, i,
                                                                          len(unlabelledloader))
            meters.update('mixup_cons_weight', mixup_consistency_weight)
            mixup_consistency_loss = mixup_consistency_weight * mixup_consistency_loss
        else:
            mixup_consistency_loss = 0
            meters.update('mixup_cons_loss', 0)

        # labeled_minibatch_size = target_var.data.ne(NO_LABEL).sum().type(torch.cuda.FloatTensor)
        # assert labeled_minibatch_size > 0

        #p_score, _ = output.topk(1, 1, True, True)
        #u_score, _ = cons_logit.topk(1, 1, True, True)
        p_score = output
        u_score = cons_logit
        gamma = 0.2
        pairwise_ranking_loss = max(0, u_score.view(-1).mean() - p_score.view(-1).mean() - gamma)

        #loss = mixup_consistency_loss
        #print(class_loss)
        #loss = pairwise_ranking_loss - 1 * mixup_consistency_loss
        #loss = 0 * pairwise_ranking_loss + 1 * mixup_consistency_loss
        loss = class_loss + 1 * mixup_consistency_loss
        #loss = class_loss + 1 * mixup_consistency_loss + 0 * pairwise_ranking_loss
        #print ('pairwise ranking loss: ', pairwise_ranking_loss)

        #print (class_loss)
        #print (mixup_consistency_loss)
        #print ("loss",loss)

        meters.update('loss', loss.item())

        prec1, prec5 = accuracy(class_logit.data, target_var.data, topk=(1, 2))

        #print ("prec1",prec1[0])
        #print ("prec5",prec5[0])
        meters.update('top1', prec1[0], minibatch_size)
        meters.update('error1', 100. - prec1[0], minibatch_size)
        meters.update('top5', prec5[0], minibatch_size)
        meters.update('error5', 100. - prec5[0], minibatch_size)

        ema_prec1, ema_prec5 = accuracy(ema_logit_labeled.data, target_var.data, topk=(1, 2))
        meters.update('ema_top1', ema_prec1[0], minibatch_size)
        meters.update('ema_error1', 100. - ema_prec1[0], minibatch_size)
        meters.update('ema_top5', ema_prec5[0], minibatch_size)
        meters.update('ema_error5', 100. - ema_prec5[0], minibatch_size)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        global_step += 1
        update_ema_variables(model, ema_model, args.ema_decay, global_step)

        # measure elapsed time
        #meters.update('batch_time', time.time() - end)
        #end = time.time()


        if i % args.print_freq == 0:
            print(
                'Epoch: [{0}][{1}/{2}]\t'
                'Class {meters[class_loss]:.4f}\t'
                'Mixup Cons {meters[mixup_cons_loss]:.4f}\t'
                'Prec@1 {meters[top1]:.3f}\t'
                'Prec@5 {meters[top5]:.3f}'.format(
                    epoch, i, len(unlabelledloader), meters=meters))
            # print ('lr:',optimizer.param_groups[0]['lr'])
    train_class_loss_list.append(meters['class_loss'].avg)
    train_error_list.append(float(meters['error1'].avg))
    train_pre_list.append(meters['top1'].avg)

NO_LABEL = -1
def validate(eval_loader, model, global_step, epoch,  ema=False, testing=False):
    class_criterion = nn.CrossEntropyLoss(reduction='sum', ignore_index=NO_LABEL).to(device)
    import utils
    import time
    meters = utils.AverageMeterSet()

    # switch to evaluate mode
    model.eval()

    output = []

    end = time.time()
    for i, (input, target) in enumerate(eval_loader):
        meters.update('data_time', time.time() - end)

        with torch.no_grad():
            input_var = torch.autograd.Variable(input.to(device))
        with torch.no_grad():
            target_var = torch.autograd.Variable(target.to(device))

        minibatch_size = len(target_var)
        #labeled_minibatch_size = target_var.data.ne(NO_LABEL).sum().type(torch.cuda.FloatTensor)
        #assert labeled_minibatch_size > 0
        #meters.update('labeled_minibatch_size', labeled_minibatch_size)

        # compute output
        output1 = model(input_var.float())
        #print ("output1",output1)
        class_loss = class_criterion(output1, target_var.long()) / minibatch_size

        output = output + list(output1.data)
        # measure accuracy and record loss
        prec1, prec5 = accuracy(output1.data, target_var.data, topk=(1, 2))
        meters.update('class_loss', class_loss.item(), minibatch_size)
        meters.update('top1', prec1[0], minibatch_size)
        meters.update('error1', 100.0 - prec1[0], minibatch_size)
        meters.update('top5', prec5[0], minibatch_size)
        meters.update('error5', 100.0 - prec5[0], minibatch_size)

        # measure elapsed time
        meters.update('batch_time', time.time() - end)
        end = time.time()

    print(' * Prec@1 {top1.avg:.3f}\tPrec@5 {top5.avg:.3f}\n'
          .format(top1=meters['top1'], top5=meters['top5']))

    val_class_loss_list.append(meters['class_loss'].avg)
    val_error_list.append(float(meters['error1'].avg))
    val_pre_list.append(meters['top1'].avg)
    return output, meters['top1'].avg

from numpy import  array
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
#values=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
#values = array(values)
#label_encoder = LabelEncoder()
#integer_encoded = label_encoder.fit_transform(values)
#print(integer_encoded)

#onehot_encoder = OneHotEncoder(sparse=False)
#integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
#onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

train_class_loss_list = []
train_error_list = []
train_pre_list = []
val_class_loss_list = []
val_error_list = []
val_pre_list = []

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def run(train_x, train_y, test_x, test_y):

    # select train test(8,9)
    #print("train_x.shape", train_x.shape)

    global train_class_loss_list
    global train_error_list
    global train_pre_list
    global val_class_loss_list
    global val_error_list
    global val_pre_list
    train_class_loss_list = []
    train_error_list = []
    train_pre_list = []
    val_class_loss_list = []
    val_error_list = []
    val_pre_list = []

    global global_step
    global_step = 0

    global best_prec1
    best_prec1 = 0

    global args
    args = parser.parse_args()

    input_size = train_x.shape[1]
    hidden_size1 = 256
    num_classes = 2

#    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = MLP(input_size, hidden_size1, num_classes).to(device)
    ema_model = MLP(input_size, hidden_size1, num_classes).to(device)
    # 0.01 for ethn, krvsk
    optimizer = pt.optim.SGD(model.parameters(), lr=1e-5, momentum=0.9, weight_decay=1e-4, nesterov =True)#
    
    args.mixup_hidden = False
    
    #optimizer = pt.optim.Adam(model.parameters(), lr=1e-5)

#    from torchvision import transforms

    train_y[train_y == -1] = 0
    test_y[test_y == -1] = 0
    train_x=torch.tensor(train_x)
    
    pos_idx = np.where(train_y == 1)[0]
    neg_idx = np.where(train_y == 0)[0]
    
    #print (pos_idx)
    #print (neg_idx)

    train_pos_X = train_x[pos_idx]
    train_neg_X = train_x[neg_idx]
    train_pos_y = train_y[pos_idx]
    train_neg_y = train_y[neg_idx]

    train_pos_y = torch.tensor(train_pos_y)
    train_neg_y = torch.tensor(train_neg_y)
    #print (train_neg_X.size())

    #n_sample_idx_u=random.sample(range(train_neg_X.shape[0]), min(train_pos_X.shape[0], int(0.3*train_neg_X.shape[0])))
    #train_pos_X = torch.cat((train_pos_X, train_neg_X[n_sample_idx_u]), 0)
    #train_pos_y = torch.cat((train_pos_y, train_neg_y[n_sample_idx_u]), 0)

    torch_dataset = torch.utils.data.TensorDataset(train_pos_X, train_pos_y)

    P_loader = torch.utils.data.DataLoader(
        # 从数据库中每次抽出batch size个样本
        dataset=torch_dataset,
        batch_size=128
    )


    torch_dataset = torch.utils.data.TensorDataset(train_neg_X, train_neg_y)

    U_loader = torch.utils.data.DataLoader(
        # 从数据库中每次抽出batch size个样本
        dataset=torch_dataset,
        batch_size=128
    )

    test_x_=torch.tensor(test_x)
    test_y=torch.tensor(test_y)
    torch_dataset = torch.utils.data.TensorDataset(test_x_, test_y)
    validloader = torch.utils.data.DataLoader(
        # 从数据库中每次抽出batch size个样本
        dataset=torch_dataset,
        batch_size=128
    )


    # fraction = 0.5 on krvskp
    # reliable negative mining
    #rn_method = 'NB'
    #rn_method = 'distance'
    #rn_method = 'SVM'
    #rn_method = 'RF'
    rn_method = 'rand'
    #n_sample_idx_u = RN_mining(train_x, train_y, rn_method, 0.5)
    n_sample_idx_u=random.sample(range(train_neg_X.shape[0]), min(train_pos_X.shape[0] * 15, int(0.4*train_neg_X.shape[0])))

    pretrain_x=train_pos_X
    pretrain_y=train_pos_y
    while len(pretrain_y) + len(train_pos_y) <= len(n_sample_idx_u):
        pretrain_x = torch.cat((pretrain_x, train_pos_X), 0)
        pretrain_y = torch.cat((pretrain_y, train_pos_y), 0)

    print (n_sample_idx_u)
    print (pretrain_x.shape)
    print (pretrain_y.shape)

    pretrain_x = torch.cat((pretrain_x, train_neg_X[n_sample_idx_u]), 0)
    pretrain_y = torch.cat((pretrain_y, train_neg_y[n_sample_idx_u]), 0)
    perm = np.random.permutation(len(pretrain_y))
    pretrain_x, pretrain_y = pretrain_x[perm], pretrain_y[perm]
    print (pretrain_y)
    #print ("pretrain_x",pretrain_x.size(),pretrain_x.size())
    torch_dataset = torch.utils.data.TensorDataset(pretrain_x, pretrain_y)

    trainloader = torch.utils.data.DataLoader(
        # 从数据库中每次抽出batch size个样本
        dataset=torch_dataset,
        batch_size=128
    )

    for epoch in range(args.start_epoch, args.epochs):
        pre_train(trainloader, model, optimizer, epoch)
        print("=========================:\n")
        prec1 = validate(validloader, model, global_step, epoch + 1)
        
        torch.cuda.empty_cache()

    train_error_list.append('====')
    val_error_list.append('====')

    #ema_model=model
    for epoch in range(args.start_epoch, args.start_epoch + 50):

        #pre_train(trainloader, model, optimizer, epoch)
        train(trainloader, U_loader, model, ema_model, optimizer, epoch)
        print("Evaluating the primary model on validation set:\n")
        output, prec1 = validate(validloader, model, global_step, epoch + 1)

        torch.cuda.empty_cache()


    train_log = OrderedDict()
    train_log['train_class_loss_list'] = train_class_loss_list
    train_log['train_error_list'] = train_error_list
    train_log['train_pre_list'] = train_pre_list

    train_log['val_pre_list'] = val_pre_list
    train_log['val_class_loss_list'] = val_class_loss_list
    train_log['val_error_list'] = val_error_list
    #print ("train")
    #print (train_class_loss_list)
    #print (train_error_list)
    #print (train_pre_list)
    #print ("val")
    #print (val_pre_list)
    #print (val_class_loss_list)
    #print (val_error_list)

    return output, train_log

x_train, y_train, x_test, y_test, prior = load_dataset('cifar10', args.labeled, args.unlabeled)
output, train_log = run(x_train, y_train, x_test, y_test)
print (train_log)
