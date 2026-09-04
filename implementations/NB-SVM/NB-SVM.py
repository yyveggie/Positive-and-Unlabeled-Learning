'''Dataset is in LIBSVM format'''

import os
import numpy as np
from Conversion import SVMlight
import random
import time

s = time.time()
path = r'data/mushroom.libsvm'
texts_1, texts_0 = SVMlight(path)
percent_p = 3  # number of folds assigned to P
texts_1 = np.array_split(texts_1, 10)  # split the class-1 samples into ten folds
texts_0 = np.array_split(texts_0, 10)  # split the class-0 samples into ten folds

for k in range(10):
    test = list(texts_1[k]) + list(texts_0[k])  # test x; one fold is taken each round
    test_y = list(len(texts_1[k]) * ['+1']) + list(len(texts_0[k]) * ['-1'])  # test y; positive is 1, negative is 0

    index_rest = sorted(set(range(10)) - set([k]))  # indices left after removing the test fold
    index_p = random.sample(index_rest, percent_p)  # randomly choose percent_p indices for P
    texts_1 = np.array(texts_1)  # convert to an array so multi-dimensional indexing works
    p = texts_1[index_p]  # the P set

    index_except_p = sorted(set(range(10)) - set([k]) - set(index_p))  # indices left after removing the test fold and P
    texts_0 = np.array(texts_0)  # convert to an array so multi-dimensional indexing works
    u = list(texts_1[index_except_p]) + list(texts_0[index_rest])  # the U set

    p = np.array([j for i in p for j in i])
    u = np.array([j for i in u for j in i])

    with open(r'D:\LPU\lpu\demo_.pos', 'w') as f:
        for i in range(len(p)):
            f.write(str(p[i]).replace(' ', '').replace('{', '').replace('}', '').replace(',', ' '))
            f.write('\n')
    with open(r'D:\LPU\lpu\demo_.unlabel', 'w') as f:
        for i in range(len(u)):
            f.write(str(u[i]).replace(' ', '').replace('{', '').replace('}', '').replace(',', ' '))
            f.write('\n')
    with open(r'D:\LPU\lpu\demo_.test', 'w') as f:
        for i in range(len(test_y)):
            f.write(str(test_y[i]))
            f.write(' ')
            f.write(str(test[i]).replace(' ', '').replace('{', '').replace('}', '').replace(',', ' '))
            f.write('\n')
    result = os.system('cd /d D:\LPU\lpu && lpu -s1 nb -s2 svm -c 1 -f demo_')
    print(result)
print("Time：", time.time() - s)