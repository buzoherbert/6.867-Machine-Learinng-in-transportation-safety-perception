#Python 3 file

from random import shuffle, randrange
import csv
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, PReLU, LeakyReLU #, Flatten
from keras.optimizers import SGD
from keras import initializers
from sklearn import preprocessing
import math
import matplotlib.pyplot as plt
from process_data import Data
from sklearn.metrics import f1_score

TEST_RATIO = 0.0 #float from 0.0 to 1.0
VAL_RATIO = 0.3 #float from 0.0 to 1.0
NORMALIZE = True #normalize data in "total_passenger_count", "total_female_count", "empty_seats", "haversine"
BATCH_SIZE = 5
INCLUDE = "ALL" #one of "trip_var", "perception_var", "contextual_var", "sociodemographic_var", "ALL"
ACTIVATION = "relu"
FILENAME = 'final_data_0.csv'

#set COMPARE to true if you want to compare our model with a "mode model" (predict everything to be the most probable perception value)
# and a "random model" (predict perception value x with probability weighted according to the distribution of perception x occuring)
COMPARE = False

data_loader = Data(VAL_RATIO, TEST_RATIO, INCLUDE, FILENAME, normalize = NORMALIZE)

title_x, title_y = data_loader.get_title()
train_x, train_y = data_loader.get_train_data()
val_x, val_y = data_loader.get_val_data()
test_x, test_y = data_loader.get_test_data()

if INCLUDE == "ALL":
    train_x = train_x[:,:-2] # remove lat and lon
    val_x = val_x[:,:-2] # remove lat and lon


trainclass_y = keras.utils.to_categorical(train_y)


M1 = 10
M2 = 10 #Set None to remove second layer
Dropout_rate = 0.0 # set 0.0 to disable dropout
input_dim = train_x.shape[1]

model = Sequential()
model.add(Dense(M1, activation=ACTIVATION, input_dim=input_dim))
if Dropout_rate > 0:
    model.add(Dropout(Dropout_rate))
if M2 != None:
    model.add(Dense(M2, activation=ACTIVATION))
    if Dropout_rate > 0:
        model.add(Dropout(Dropout_rate))
model.add(Dense(5))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])

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
        total += (Y_true[i][0] - Y_pred[i])**2
    return total

train_rates = []
val_rates = []
test_rates = []

train_f1 = []
val_f1 = []
test_f1 = []

for i in range(50):
    model.fit(train_x, trainclass_y, epochs=1, batch_size=BATCH_SIZE)
    predict_train = model.predict_classes(train_x, verbose=0)
    train_rates.append(get_error_rate(train_y, predict_train))
    train_f1.append(f1_score(train_y, predict_train, average='weighted'))
    predict_val = model.predict_classes(val_x, verbose=0)
    val_rates.append(get_error_rate(val_y, predict_val))
    val_f1.append(f1_score(val_y, predict_val, average='weighted'))
    if len(test_y) > 0:
        predict_test = model.predict_classes(test_x, verbose=0)
        test_rates.append(get_error_rate(test_y, predict_test))
        test_f1.append(f1_score(test_y, predict_test, average='weighted'))

def get_confusion_mtx(y_true, y_pred):
    confusion = np.zeros((5, 5))
    for i in range(len(y_true)):
        confusion[y_true[i][0]][y_pred[i]] += 1
    return confusion


confusion_train = get_confusion_mtx(train_y, predict_train)
print("Confusion matrix for training data:")
print(confusion_train)

confusion_val = get_confusion_mtx(val_y, predict_val)
print("Confusion matrix for validation data:")
print(confusion_val)

if len(test_y) > 0:
    confusion_test = get_confusion_mtx(test_y, predict_test)
    print("Confusion matrix for test data:")
    print(confusion_test)


error = val_rates[-1]

print("final error rate on val data: ", error)

f1 = val_f1[-1]

print("final weighted f1 rate on val data: ", f1)

def weighted_random(weights):
    x = randrange(sum(weights))
    for i in range(len(weights)):
        if x < weights[i]:
            return i
        else:
            x = x - weights[i]

if COMPARE:

    sums = np.sum(trainclass_y, axis=0)
    mode = max(range(5), key=lambda x: sums[x])
    mode_prediction = np.array([mode for i in range(val_y.size)])
    random_pred = np.array([weighted_random(sums) for i in range(val_y.size)])
    error_mode = get_error_rate(val_y, mode_prediction)
    error_random = get_error_rate(val_y, random_pred)

    confusion_mode = get_confusion_mtx(val_y, mode_prediction)
    confusion_rand = get_confusion_mtx(val_y, random_pred)

    print("error rate using mode: ", error_mode)
    print("f1 rate using mode: ", f1_score(val_y, mode_prediction, average='weighted'))

    print("error rate using random: ", error_random)
    print("f1 rate using random: ", f1_score(val_y, random_pred, average='weighted'))

    print("Confusion matrix for mode:")
    print(confusion_mode)

    print("Confusion matrix for random:")
    print(confusion_rand)

plt.plot(train_rates, 'go-', label="train")
plt.plot(val_rates, 'ro-', label="val")
plt.plot(test_rates, 'bo-', label="test")
plt.title("error rate vs epoch number")
plt.legend()
plt.show()

plt.plot(train_f1, 'go-', label="train")
plt.plot(val_f1, 'ro-', label="val")
plt.plot(test_f1, 'bo-', label="test")
plt.title("weighted f1 score vs epoch number")
plt.legend()
plt.show()
