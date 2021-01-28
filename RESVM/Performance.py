from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import numpy as np

f1 = []
accuracy = []

for k in range(10):
    with open(r'C:\Users\yyveggie\Desktop\python\PU_Learning\RESVM\predictions_{}.txt'.format(str(k)), 'r') as f:
        pred_y = f.readlines()
    y_pred = [int(i.split(' ')[0]) for i in pred_y]

    with open(r'C:\Users\yyveggie\Desktop\python\PU_Learning\RESVM\data\test_{}.libsvm'.format(str(k)), 'r') as f:
        true_y = f.readlines()
    y_true = [int(i.split(' ')[0]) for i in true_y]

    f1.append(f1_score(y_true, y_pred, average='binary'))
    accuracy.append(accuracy_score(y_true, y_pred))

print("F1-Score：", np.mean(np.array(f1)))
print("Accuracy：", np.mean(np.array(accuracy)))