import pandas as pd
from pubp_nn import PUNN
from model import *
from train_nnPU import *
from Conversion import CSV
import chainer.cuda
import chainer

def experiment():
    percent_p = 3
    path = r'C:\Users\yyveggie\Desktop\UCI\Conversion\mushroom.csv'
    ite = 10
    epoch = 100
    batchsize = 1000

    seed = 2020

    gpu = True

    acc_nnpu = []
    f1_nnpu = []
    acc_nnpusb = []
    f1_nnpusb = []

    for k in range(ite):
        np.random.seed(seed)
        #PN classification
        texts_1, texts_0 = CSV(path)
        texts_1 = np.array_split(texts_1, 10)  # 将类别为1的样本集分成十份
        texts_0 = np.array_split(texts_0, 10)  # 将类别为0的样本集分成十份
        x_test = np.array(list(texts_1[k]) + list(texts_0[k]))  # 测试集x，每一份每轮选一次
        t_test = np.array(list(len(texts_1[k]) * [1]) + list(len(texts_0[k]) * [0]))  # 测试集y，正例为1，负例为0

        index_rest = sorted(set(range(10)) - set([k]))
        texts_1 = np.array(texts_1)
        texts_0 = np.array(texts_0)
        texts_1 = np.array([j for i in texts_1[index_rest] for j in i])
        texts_0 = np.array([j for i in texts_0[index_rest] for j in i])
        x = np.vstack((texts_1, texts_0))
        t = pd.Series([1] * len(texts_1) + [0] * len(texts_0))
        dim = x.shape[1]
        print(x.shape)

        x_train = x
        t_train = pd.Series([1] * len(texts_1) + [0] * len(texts_0))

        pi = np.mean(t_train)

        model = MultiLayerPerceptron(dim)
        optimizer = optimizers.Adam(1e-5)
        optimizer.setup(model)

        if gpu:
            gpu_device = 0
            cuda.get_device(gpu_device).use()
            model.to_gpu(gpu_device)
            xp = cuda.cupy
        else:
            xp = np

        model, optimizer = train(x, t, epoch, model, optimizer, batchsize, xp)

        x_p = x_train[t_train==1]

        xp_prob = np.array([])
        for j in six.moves.range(0, len(x_p), batchsize):
            X = Variable(xp.array(x_p[j:j + batchsize], xp.float32))
            g = chainer.cuda.to_cpu(model(X).data).T[0]
            xp_prob = np.append(xp_prob, 1/(1+np.exp(-g)), axis=0)
        xp_prob /= np.mean(xp_prob)
        xp_prob = xp_prob
        xp_prob /= np.max(xp_prob)
        print(xp_prob)
        rand = np.random.uniform(size=len(x_p))
        x_p = x_p[xp_prob > rand]
        perm = np.random.permutation(len(x_p))
        pdata = int(percent_p / 10 * len(x))  # p样本数量，占了总数的3/10
        x_p = x_p[perm[:pdata]]

        tp = np.ones(len(x_p))
        tu = np.zeros(len(x_train))
        t_train = np.concatenate([tp, tu], axis=0)

        x_train = np.concatenate([x_p, x_train], axis=0)

        print(x_train.shape)
        print(t_train.shape)
        print(x_test.shape)
        print(t_test.shape)

        model = MultiLayerPerceptron(dim)
        optimizer = optimizers.Adam(alpha=1e-5)
        optimizer.setup(model)
        optimizer.add_hook(chainer.optimizer.WeightDecay(0.005))

        if gpu:
            gpu_device = 0
            cuda.get_device(gpu_device).use()
            model.to_gpu(gpu_device)
            xp = cuda.cupy
        else:
            xp = np

        model, optimizer, acc1, acc2, f1_binary1, f1_binary2 = train_pu(x_train, t_train, x_test, t_test, pi, epoch, model, optimizer, batchsize, xp)

        acc_nnpu.append(acc1)
        f1_nnpu.append(f1_binary1)
        acc_nnpusb.append(acc2)
        f1_nnpusb.append(f1_binary2)

        seed += 1

        print("Iter：", k)
        print("acc_nnpu：", acc1)
        print("acc_nnpusb：", acc2)
        print("f1_nnpu：", f1_binary1)
        print("f1_nnpusb：", f1_binary2)

    print("acc_nnpu_mean：", np.mean(acc_nnpu))
    print("f1_nnpu_mean：", np.mean(f1_nnpu))
    print("acc_nnpusb_mean：", np.mean(acc_nnpusb))
    print("f1_nnpusb_mean：", np.mean(f1_nnpusb))

if __name__ == "__main__":
    experiment()
