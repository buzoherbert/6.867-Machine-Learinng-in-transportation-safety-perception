#Python 3 file

from random import shuffle
import csv
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout #, Flatten
#from keras.optimizers import SGD
from keras import initializers
from keras import losses
import math
import matplotlib.pyplot as plt
from process_data import Data
from sklearn.metrics import f1_score

TEST_RATIO = 0.0 #float from 0.0 to 1.0
VAL_RATIO = 0.3 #float from 0.0 to 1.0
NORMALIZE = True #normalize data in "total_passenger_count", "total_female_count", "empty_seats", "haversine"
BATCH_SIZE = 5
INCLUDE = "ALL" #one of "trip_var", "perception_var", "contextual_var", "sociodemographic_var", "ALL"

data_loader = Data(VAL_RATIO, TEST_RATIO, INCLUDE, normalize = NORMALIZE)

title_x, title_y = data_loader.get_title()
train_x, train_y = data_loader.get_train_data()
val_x, val_y = data_loader.get_val_data()
test_x, test_y = data_loader.get_test_data()

#trainclass_y = keras.utils.to_categorical(train_y)


M1 = 10
M2 = 10 #Set None to remove second layer
Dropout_rate = 0.3 # set 0.0 to disable dropout
input_dim = len(title_x)

model = Sequential()
model.add(Dense(M1, activation='relu', input_dim=input_dim))
if Dropout_rate > 0:
    model.add(Dropout(Dropout_rate))
if M2 != None:
    model.add(Dense(M2, activation='relu'))
    if Dropout_rate > 0:
        model.add(Dropout(Dropout_rate))
model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer="adam")

"""
model.fit(final_train, trainclass_y, epochs=50, batch_size=5)
predict_train = model.predict_classes(final_train, verbose=0)
predict_val = model.predict_classes(final_val, verbose=0)
if len(test_y) > 0:
    predict_test = model.predict_classes(final_test, verbose=0)
"""

def get_error_rate(Y_true, Y_pred):
    n = Y_true.size
    err = 0
    for i in range(n):
        if Y_true[i][0] != Y_pred[i]:
            err += 1
    return float(err)/n

def sse(Y_true, Y_pred):
    n = Y_true.size
    total = 0
    for i in range(n):
        total += (Y_true[i][0] - Y_pred[i][0])**2
    return total

train_rates = []
val_rates = []
test_rates = []

for i in range(50):
    model.fit(train_x, train_y, epochs=1, batch_size=BATCH_SIZE)
    predict_train = model.predict(train_x, verbose=0)
    train_rates.append(sse(train_y, predict_train))
    predict_val = model.predict(val_x, verbose=0)
    val_rates.append(sse(val_y, predict_val))
    if len(test_y) > 0:
        predict_test = model.predict_classes(test_x, verbose=0)
        test_rates.append(sse(test_y, predict_test))

sse_final = val_rates[-1]

print("final sse for val data: ", sse_final)

pred_mean = np.mean(train_y)
sst = sse(val_y, np.array([[pred_mean] for i in range(val_y.size)]))

print("sst for val data: ", sst)

osr2 = 1.0 - sse_final/sst

print("osr2: ", osr2)


def get_confusion_mtx(y_true, y_pred):
    confusion = np.zeros((5, 5))
    for i in range(len(y_true)):
        confusion[y_true[i][0]][y_pred[i]] += 1
    return confusion

predict_val_int = predict_val.round().astype(int)
confusion_val = get_confusion_mtx(val_y, predict_val_int)
print("Confusion matrix for validation data:")
print(confusion_val)
print(get_error_rate(val_y, predict_val_int))
print(f1_score(val_y, predict_val_int, average='weighted'))


plt.plot(train_rates, 'go-', label="train")
plt.plot(val_rates, 'ro-', label="val")
plt.plot(test_rates, 'bo-', label="test")
plt.title("mean squared error vs epoch number")
plt.legend()
plt.show()
