''' LIBSVM data format '''

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from pusb_linear_kernel import PU
from densratio import densratio
from Conversion import CSV

def experiment():
    percent_p = 3
    path = r'data/mushroom.csv'
    seed = 2019

    est_error_upu = []
    est_binary_upu = []
    est_binary_pusb = []
    est_error_pusb = []
    est_error_drsb = []
    est_binary_drsb = []

    for k in range(10):
        np.random.seed(seed)
        pi = 0.6        # class prior, the share of P within U; whether it should be computed from the ratios below rather than fixed in advance is open
        classifier = LogisticRegression(C=0.01, penalty='l2', solver='liblinear')
        texts_1, texts_0 = CSV(path)
        texts_1 = np.array_split(texts_1, 10)  # split the class-1 samples into ten folds
        texts_0 = np.array_split(texts_0, 10)  # split the class-0 samples into ten folds
        x_test = np.array(list(texts_1[k]) + list(texts_0[k]))  # test x; one fold is taken each round
        t_test = np.array(list(len(texts_1[k]) * [1]) + list(len(texts_0[k]) * [0]))  # test y; positive is 1, negative is 0

        index_rest = sorted(set(range(10)) - set([k]))  # indices left after the test set
        texts_1 = np.array(texts_1)
        texts_0 = np.array(texts_0)
        texts_1 = np.array([j for i in texts_1[index_rest] for j in i])     # p
        texts_0 = np.array([j for i in texts_0[index_rest] for j in i])     # n
        x = np.vstack((texts_1, texts_0))   # P and N form the training set
        one = np.ones((len(x), 1))  #
        x_pn = np.concatenate([x, one], axis=1)
        t = pd.Series([1] * len(texts_1) + [0] * len(texts_0))
        classifier.fit(x_pn, t)

        x_train = x
        t_train = t

        xp = x_train[t_train == 1]
        one = np.ones((len(xp), 1))
        xp_temp = np.concatenate([xp, one], axis=1)
        xp_prob = classifier.predict_proba(xp_temp)[:, 1]
        # xp_prob /= np.mean(xp_prob)
        xp_prob = xp_prob ** 20
        xp_prob /= np.max(xp_prob)
        rand = np.random.uniform(size=len(xp))
        temp = xp[xp_prob > rand]
        pdata = int(percent_p / 10 * len(x))  # size of P, three tenths of the total
        while (len(temp) < pdata):
            rand = np.random.uniform(size=len(xp))
            temp = np.concatenate([temp, xp[xp_prob > rand]], axis=0)
        xp = temp
        perm = np.random.permutation(len(xp))
        xp = xp[perm[:pdata]]
        u = int(6 / 10 * len(x))  # size of U, six tenths of the total
        updata = np.int(u * pi)   # positives in U = |U| * class prior
        undata = u - updata       # negatives in U = |U| - positives in U

        xp_temp = x_train[t_train == 1]
        xn_temp = x_train[t_train == 0]
        perm = np.random.permutation(len(xp_temp))
        xp_temp = xp_temp[perm[:updata]]

        perm = np.random.permutation(len(xn_temp))
        xn_temp = xn_temp[perm[:undata]]
        xu = np.concatenate([xp_temp, xn_temp], axis=0)

        x = np.concatenate([xp, xu], axis=0)

        tp = np.ones(len(xp))
        tu = np.zeros(len(xu))
        t = np.concatenate([tp, tu], axis=0)

        updata = np.int(1000 * pi)
        undata = 1000 - updata

        xp_test = x_test[t_test == 1]
        perm = np.random.permutation(len(xp_test))
        xp_test = xp_test[perm[:updata]]
        xn_test = x_test[t_test == 0]
        perm = np.random.permutation(len(xn_test))
        xn_test = xn_test[perm[:undata]]

        x_test = np.concatenate([xp_test, xn_test], axis=0)
        tp = np.ones(len(xp_test))
        tu = np.zeros(len(xn_test))
        t_test = np.concatenate([tp, tu], axis=0)

        pu = PU(pi=pi)
        x_train = x
        res, x_test_kernel = pu.optimize(x, t, x_test)
        acc1, f1_binary1 = pu.test(x_test_kernel, res, t_test, quant=False)
        acc2, f1_binary2 = pu.test(x_test_kernel, res, t_test, quant=True, pi=pi)

        result = densratio(x_train[t == 1], x_train[t == 0])
        r = result.compute_density_ratio(x_test)
        temp = np.copy(r)
        temp = np.sort(temp)
        theta = temp[np.int(np.floor(len(x_test) * (1 - pi)))]
        pred = np.zeros(len(x_test))
        pred[r > theta] = 1
        acc3 = np.mean(pred == t_test)
        f1_binary3 = f1_score(t_test, pred, average='binary')

        est_error_upu.append(acc1)
        est_binary_upu.append(f1_binary1)
        est_error_pusb.append(acc2)
        est_binary_pusb.append(f1_binary2)
        est_error_drsb.append(acc3)
        est_binary_drsb.append(f1_binary3)

        seed += 1

        print("Iter：", k)
        print("upu_accuracy ", acc1)
        print("upu_f1_binary ", f1_binary1)
        print("pusb_accuracy ", acc2)
        print("pusb_f1_binary ", f1_binary2)
        print("drsb_accuracy ", acc3)
        print("drsb_f1_binary ", f1_binary3)

    print("Accuracy for uPU：", np.mean(est_error_upu))
    print("F1-Score for uPU：", np.mean(est_binary_upu))
    print("Accuracy for PUSB：", np.mean(est_error_pusb))
    print("F1-Score for PUSB：", np.mean(est_binary_pusb))
    print("Accuracy for DRSB：", np.mean(est_error_drsb))
    print("F1-Score for DRSB：", np.mean(est_binary_drsb))

experiment()