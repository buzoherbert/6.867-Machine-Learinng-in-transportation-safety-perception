from sklearn.metrics import f1_score
import numpy as np


FLIP = False

# confusion_mtx[i][j] represents number of people with true safety perception i, but model predicted a safety perception of j
# NOTE: if data is flipped (i.e. j is true value and i is predicted value), set FLIP to True
confusion_mtx = [[11, 7, 15, 12, 7],
                 [4, 6, 8, 5, 2],
                 [16, 9, 46, 23, 5],
                 [12, 9, 24, 21, 12],
                 [4, 0, 3, 9, 7]]


mtx = np.array(confusion_mtx)
if FLIP:
    mtx = mtx.transpose()

y_true = []
y_pred = []

num_total = 0
num_true = 0

for i in range(5):
    for j in range(5):
        if i==j:
            num_true += mtx[i][j]
        num_total += mtx[i][j]
        
        for x in range(mtx[i][j]):
            y_true.append(i)
            y_pred.append(j)

f1 = f1_score(y_true, y_pred, average='weighted')
acc = num_true/num_total

print("accuracy is: ", acc)
print("weighted f1 score is: ", f1)
