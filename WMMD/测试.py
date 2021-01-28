#!/usr/bin/env python
# coding: utf-8

# base modules
import numpy as np
from sklearn.metrics import f1_score, accuracy_score

# torch modules
from torch.utils.data import DataLoader

# customized module
from wmmd import WMMD
from data import generate_normal_PU_tr_data, generate_normal_te_data, SplitData
from Conversion import CSV

score_list_wmmd, pred_list_wmmd  = [], []
pi_plus = 0.5   # 类先验,u中正例的比例，也可以说是总体样本中我们通过先验知识得到的正例的比例
                # 因为我们并不知道总体分布， 所以我们假设正负各占50%
total_f1 = []
total_acc = []
# n_p = 50    # p的数量
# n_u = 500   # u的数量
# n_te = 1000   # 测试集数量
batch_size = 10000  # Mini batch size
device = 'cpu'  # 计算方式
path = r'C:\Users\yyveggie\Desktop\UCI\Conversion\mushroom.csv'
texts_1, texts_0 = CSV(path)
# Selecting optimal hyper-parameters
hyper_wmmd_optimal = [0, 0, 10] # threshold, gamma, val
gamma_list = [1, 0.4, 0.2, 0.1, 0.05]   # list of hyperparameter gamma
for k in range(10):
    k_f1, k_acc = [], []
    print("Iter：", k)
    # data_pu_tr = generate_normal_PU_tr_data(pi_plus, n_p, n_u)  # 训练集
    # data_te = generate_normal_te_data(pi_plus, n_te)    # 测试集
    data_pu_tr, data_te = SplitData(k, texts_1, texts_0)
    # DataLoader是Pytorch中用来处理模型输入数据的一个工具类。
    # 通过使用DataLoader，我们可以方便地对数据进行相关操作，
    # 比如我们可以很方便地设置batch_size，对于每一个epoch是否随机打乱数据，是否使用多线程等等。
    generator_te = DataLoader(np.hstack((data_te['X'], data_te['PN_label'].reshape(-1,1))),
                                          batch_size=batch_size, shuffle=False)
    # Selecting optimal hyper-parameters
    for gam in gamma_list:
        model_wmmd = WMMD(data_dict=data_pu_tr, device=device, pi_plus=pi_plus,
                          dict_parameter={'gamma': gam})
        model_wmmd.calculate_val_risk()
        # print('n_u:{}, val_loss:{}, gamma:{}'.format(n_u, model_wmmd.val_loss, gam))
        if hyper_wmmd_optimal[2] > model_wmmd.val_loss:
            hyper_wmmd_optimal[0] = model_wmmd.threshold
            hyper_wmmd_optimal[1] = gam
            hyper_wmmd_optimal[2] = model_wmmd.val_loss
    #Training
    model_wmmd = WMMD(data_dict = data_pu_tr, device=device, pi_plus=pi_plus,
                      dict_parameter={'gamma':hyper_wmmd_optimal[1]})
    #Test
    for data_te_batch in generator_te:
        X_te_batch, Y_te_batch = data_te_batch[:,:-1], data_te_batch[:,-1]
        X_te_batch = (X_te_batch).to(device)

        Y_te = Y_te_batch.data.numpy()

        score_wmmd = model_wmmd.score(X_te_batch).data.cpu().numpy().ravel()
        pred_wmmd = (2*((score_wmmd < hyper_wmmd_optimal[0])+0.0) -1)
        f1 = f1_score(Y_te, pred_wmmd)
        acc = accuracy_score(Y_te, pred_wmmd)
        k_f1.append(f1)
        k_acc.append(acc)
        total_f1.append(f1)
        total_acc.append(acc)
        del X_te_batch
    print("current f1-score:", np.mean(np.array(k_f1)))
    print("current accuracy:", np.mean(np.array(k_acc)))

print("Mean of F1-Score:", np.mean(np.array(total_f1)))
print("Mean of Accuracy:", np.mean(np.array(total_acc)))