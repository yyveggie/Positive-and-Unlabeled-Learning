
import os, random
import numpy as np
from automl import validate
from util import log
from read import read_df
from preprocess import clean_df, feature_engineer
from util import timeit
import numpy as np
import copy

from mixpul import run
#from mixpu import copu

@timeit
def get_cur_label(train_data, train_ground_truth, cur_idx):
    cur_y = copy.copy(train_ground_truth)
    cur_y.loc[cur_y['label'] == 1] = 0
    cur_y.loc[cur_idx, "label"] = 1
    y = cur_y.loc[:, "label"]
    return train_data, y
    #neg_idx = y.index[y == 0]
#    neg_samp_num = np.min(np.array([len(cur_idx), len(y) - len(cur_idx)]))
#    neg_idx_samp = random.sample(range(len(neg_idx)), neg_samp_num)
#    get_neg_idx = np.array(neg_idx[neg_idx_samp].values)
    #get_neg_idx = np.array(neg_idx.values)
    #train_idx = np.array(list(cur_idx) + list(get_neg_idx))
    #cur_y = cur_y.loc[train_idx, "label"]
    #cur_train_data = train_data.loc[train_idx, :]
    #return cur_train_data, cur_y, get_neg_idx


def run_upu():
    pos_per = [1, 5, 10, 20, 40]
    #pos_per = [1, 5, 10, 20, 40, 60, 80, 100]
    datasets = ["titanic", "ethn", "krvskp", "mushroom", "sat",  "spambase", "texture", "twonorm"]#, "zhihu", "luoji", "myhug", "kaiyan", "nip", "yjp", "ttgwm"]
#    datasets = ["zhihu", "luoji", "myhug", "kaiyan", "nip", "yjp", "ttgwm"]

    for dataset in datasets:

        schema = "data/" + dataset + "/schmea"
        train_data_path = "data/" + dataset + "/train.data"
        train_label_path = "data/" + dataset + "/train.solution"
        test_data_path = "data/" + dataset + "/test.data"
        test_label_path = "data/" + dataset + "/test.solution"
        results = open("pr_results/" + dataset, "w")

        ########read_df############
        train_data = read_df(train_data_path, schema)
        train_ground_truth = read_df(train_label_path, "")
        clean_df(train_data)
        feature_engineer(train_data)
        test_data = read_df(test_data_path, schema)
        test_label = read_df(test_label_path, "")
        test_label = test_label.loc[:, "label"]
        clean_df(test_data)
        feature_engineer(test_data)
        ########read_df############

        test_data = test_data.values

        for per in pos_per:
            pos_index_path = "data/" + dataset + "/pos_percent" + str(per) + ".npy"
            print("positive perecnt " + str(per))
            results.write("positive perecnt " + str(per) + ":\n")
            pos_idx = np.loadtxt(pos_index_path)
            pos_idx = pos_idx.astype(int)
            get_score = list()

            for i in range(len(pos_idx)):
                cur_idx = pos_idx[i]

                cur_train_data, cur_train_label = get_cur_label(train_data, train_ground_truth, cur_idx)

                cur_train_data = cur_train_data.values
                cur_train_label = cur_train_label.values
                # print (cur_train_data)

                # ==================
                # INSERT YOUR CODE HERE
                y_h, train_log = run(cur_train_data, cur_train_label, test_data, test_label)
                log_path = "pr_log/" + dataset + "_pos_percent" + str(per) + "_trial" +  str(i) + ".txt"
                with open(log_path, 'w') as writer:
                    tr = '[' +  ','.join([str(x) for x in train_log['train_error_list']]) + ']'
                    val = '[' + ','.join([str(x) for x in train_log['val_error_list']]) + ']'
                    writer.write(tr)
                    writer.write('\n\n\n')
                    writer.write(val)

                y_h = [a.tolist()[1] for a in y_h]
                #print (type(y_h))
                #print (y_h)
                y_h = np.array(y_h)
                #y_h = copu(cur_train_data, cur_train_label, test_data)

                # ==================

                if os.path.exists(test_label_path):
                    score = validate(y_h, test_label_path)
                    get_score.append(score)
                    log(f"score = {score}")
            avg = np.mean(np.array(get_score))
            std = np.mean(np.std(get_score))
            results.write("avg: " + str(avg) + "\n")
            results.write("std: " + str(std) + "\n")

if __name__ == "__main__":
    run_upu()
