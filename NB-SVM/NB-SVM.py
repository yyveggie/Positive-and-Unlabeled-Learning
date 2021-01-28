'''数据集是Libsvm格式'''

import os
import numpy as np
from Conversion import SVMlight
import random
import time

s = time.time()
path = r'C:\Users\yyveggie\Desktop\UCI\Conversion\mushroom.libsvm'
texts_1, texts_0 = SVMlight(path)
percent_p = 3  # p所占份数
texts_1 = np.array_split(texts_1, 10)  # 将类别为1的样本集分成十份
texts_0 = np.array_split(texts_0, 10)  # 将类别为0的样本集分成十份

for k in range(10):
    test = list(texts_1[k]) + list(texts_0[k])  # 测试集x，每一份每轮选一次
    test_y = list(len(texts_1[k]) * ['+1']) + list(len(texts_0[k]) * ['-1'])  # 测试集y，正例为1，负例为0

    index_rest = sorted(set(range(10)) - set([k]))  # 除去测试集剩余索引
    index_p = random.sample(index_rest, percent_p)  # 随机选择percent_p个p索引
    texts_1 = np.array(texts_1)  # 转换为array格式可进行多维索引
    p = texts_1[index_p]  # p集合

    index_except_p = sorted(set(range(10)) - set([k]) - set(index_p))  # 除去测试集和p集合剩余索引
    texts_0 = np.array(texts_0)  # 转换为array格式可进行多维索引
    u = list(texts_1[index_except_p]) + list(texts_0[index_rest])  # u集合

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