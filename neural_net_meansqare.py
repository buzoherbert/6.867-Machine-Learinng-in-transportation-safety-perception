#Python 3 file

import csv
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout #, Flatten
from keras.regularizers import l2
#from keras.optimizers import SGD
from keras import initializers
from keras import losses
import math
import matplotlib.pyplot as plt
from process_data import Data
from sklearn.metrics import f1_score

TEST_RATIO = 0.2 #float from 0.0 to 1.0
VAL_RATIO = 0.2 #float from 0.0 to 1.0
NORMALIZE = True #normalize data in "total_passenger_count", "total_female_count", "empty_seats", "haversine"
BATCH_SIZE = 5
INCLUDE = "ALL" #one of "trip_var", "instant_var", "perception_var", "contextual_var", "sociodemographic_var", "ALL"
FILENAMES = ['final_data_0.csv','final_data_1.csv','final_data_2.csv','final_data_3.csv','final_data_4.csv']
NUM_EPOCHS = 50
M1 = 10
M2 = 10 #Set None to remove second layer
Dropout_rates = [0.3, 0.2, 0.1, 0.0] #[0.0, 0.1, 0.2, 0.3] # set 0.0 to disable dropout
Regularizations = [0, 0.01, 0.1] #[0, 0.001, 0.01, 0.1]
DDOF = 0

SHOW_CONFUSIONS = False
SHOWPLOTS = False

train_rates = []
val_rates = []
test_rates = []

train_f1 = []
val_f1 = []
test_f1 = []

all_train_x, all_train_y, all_val_x, all_val_y, all_test_x, all_test_y = [], [], [], [], [], []

models = []

for filename in FILENAMES:
    data_loader = Data(VAL_RATIO, TEST_RATIO, INCLUDE, filename, normalize = NORMALIZE)

    title_x, title_y = data_loader.get_title()
    train_x, train_y = data_loader.get_train_data()
    val_x, val_y = data_loader.get_val_data()
    test_x, test_y = data_loader.get_test_data()

    train_x = train_x[:,:-2] # remove lat and lon
    val_x = val_x[:,:-2] # remove lat and lon
    test_x = test_x[:,:-2]

    all_train_x.append(train_x)
    all_train_y.append(train_y)
    all_val_x.append(val_x)
    all_val_y.append(val_y)
    all_test_x.append(test_x)
    all_test_y.append(test_y)

class Model:
    def __init__(self, M1, M2, dropout_rate, input_dim, regularization):
        model = Sequential()
        if regularization == 0:
            model.add(Dense(M1, activation='relu', input_dim=input_dim))
        else:
            model.add(Dense(M1, activation='relu', input_dim=input_dim, kernel_regularizer=l2(regularization)))
        if dropout_rate > 0:
            model.add(Dropout(dropout_rate))
        if M2 != None:
            if regularization == 0:
                model.add(Dense(M2, activation='relu'))
            else:
                model.add(Dense(M2, activation='relu', kernel_regularizer=l2(regularization)))
            if dropout_rate > 0:
                model.add(Dropout(dropout_rate))
        model.add(Dense(1))

        model.compile(loss='mean_squared_error', optimizer="adam")
        self.model = model

    def train(self, train_x, train_y, val_x, val_y, test_x, test_y, num_epochs):
        
        train_rates = []
        val_rates = []
        test_rates = []

        train_osr2 = []
        val_osr2 = []
        test_osr2 = []

        model = self.model
        for i in range(num_epochs):
            model.fit(train_x, train_y, epochs=1, batch_size=BATCH_SIZE, verbose=0)
            predict_train = model.predict(train_x, verbose=0)
            train_rates.append(sse(train_y, predict_train))
            train_osr2.append(osr2(train_y, predict_train, train_y))
            predict_val = model.predict(val_x, verbose=0)
            val_rates.append(sse(val_y, predict_val))
            val_osr2.append(osr2(val_y, predict_val, train_y))
            if len(test_y) > 0:
                predict_test = model.predict(test_x, verbose=0)
                test_rates.append(sse(test_y, predict_test))
                test_osr2.append(osr2(test_y, predict_test, train_y))

        predict_train_int = np.clip(predict_train.round().astype(int), 0, 4)
        predict_val_int = np.clip(predict_val.round().astype(int), 0, 4)
        predict_test_int = np.clip(predict_test.round().astype(int), 0, 4)

        return train_rates, val_rates, test_rates, train_osr2, val_osr2, test_osr2, predict_train_int, predict_val_int, predict_test_int, model

def get_acc_rate(Y_true, Y_pred):
    n = Y_true.size
    err = 0
    for i in range(n):
        if Y_true[i][0] != Y_pred[i]:
            err += 1
    return 1.0 - float(err)/n

def sse(Y_true, Y_pred):
    n = Y_true.size
    total = 0
    for i in range(n):
        total += (Y_true[i][0] - Y_pred[i][0])**2
    return total

def osr2(Y_true, Y_pred, train_y):
    sse_val = sse(Y_true, Y_pred)
    pred_mean = np.mean(train_y)
    sst = sse(Y_true, np.array([[pred_mean] for i in range(Y_true.size)]))
    osr2 = 1.0 - sse_val/sst
    return osr2


def get_confusion_mtx(y_true, y_pred):
    confusion = np.zeros((5, 5))
    for i in range(len(y_true)):
        confusion[y_true[i][0]][y_pred[i]] += 1
    return confusion

def get_total_mtx(y_trues, y_preds):
    confusions = []
    for i in range(len(y_trues)):
        confusions.append(get_confusion_mtx(y_trues[i], y_preds[i]))
    total = np.sum(np.stack(confusions), axis=0)
    return total/total.sum()

for dr in Dropout_rates:
    for regul in Regularizations:
        train_rates, val_rates, test_rates, train_osr2, val_osr2, test_osr2, predict_train, predict_val, predict_test, models = [], [], [], [], [], [], [], [], [], []
        for i in range(len(FILENAMES)):
            input_dim = all_train_x[i].shape[1]
            model_class = Model(M1, M2, dr, input_dim, regul)
            data = model_class.train(all_train_x[i], all_train_y[i], all_val_x[i], all_val_y[i], all_test_x[i], all_test_y[i], NUM_EPOCHS)
            train_rates.append(np.array(data[0]))
            val_rates.append(np.array(data[1]))
            test_rates.append(np.array(data[2]))
            train_osr2.append(np.array(data[3]))
            val_osr2.append(np.array(data[4]))
            test_osr2.append(np.array(data[5]))
            predict_train.append(data[6])
            predict_val.append(data[7])
            predict_test.append(data[8])
            models.append(data[9])

        final_train_rates = np.stack(train_rates)
        final_val_rates = np.stack(val_rates)
        final_test_rates = np.stack(test_rates)
        final_train_osr2 = np.stack(train_osr2)
        final_val_osr2 = np.stack(val_osr2)
        final_test_osr2 = np.stack(test_osr2)

        print("Dropout ", dr, ", regul ", regul, ":")
        print("final val rates (mean, stderr): ", np.mean(final_val_rates[:, -1]), ", ", np.std(final_val_rates[:, -1], ddof=DDOF))
        print("final val osr2 (mean, stderr): ", np.mean(final_val_osr2[:, -1]), ", ", np.std(final_val_osr2[:, -1], ddof=DDOF))
        
        #print("final test rates (mean, stderr): ", np.mean(final_test_rates[:, -1]), ", ", np.std(final_test_rates[:, -1], ddof=DDOF))
        #print("final test osr2 (mean, stderr): ", np.mean(final_test_osr2[:, -1]), ", ", np.std(final_test_osr2[:, -1], ddof=DDOF))
        
        #print("final train rates (mean, stderr): ", np.mean(final_train_rates[:, -1]), ", ", np.std(final_train_rates[:, -1], ddof=DDOF))
        #print("final train osr2 (mean, stderr): ", np.mean(final_train_osr2[:, -1]), ", ", np.std(final_train_osr2[:, -1], ddof=DDOF))

        if SHOW_CONFUSIONS:
            confusion_train = get_total_mtx(all_train_y, predict_train)
            print("Confusion matrix for training data:")
            print(confusion_train)

            confusion_val = get_total_mtx(all_val_y, predict_val)
            print("Confusion matrix for validation data:")
            print(confusion_val)
            
            confusion_test = get_total_mtx(all_test_y, predict_test)
            print("Confusion matrix for test data:")
            print(confusion_test)

        if SHOWPLOTS:

            plt.plot(np.mean(final_train_rates, axis=0), 'go-', label="train")
            plt.plot(np.mean(final_val_rates, axis=0), 'ro-', label="val")
            plt.plot(np.mean(final_test_rates, axis=0), 'bo-', label="test")
            plt.title("average sse vs epoch number")
            plt.legend()
            plt.show()

            plt.plot(np.mean(final_train_osr2, axis=0), 'go-', label="train")
            plt.plot(np.mean(final_val_osr2, axis=0), 'ro-', label="val")
            plt.plot(np.mean(final_test_osr2, axis=0), 'bo-', label="test")
            plt.title("average osr2 vs epoch number")
            plt.legend()
            plt.show()

"""
model.fit(final_train, trainclass_y, epochs=50, batch_size=5)
predict_train = model.predict_classes(final_train, verbose=0)
predict_val = model.predict_classes(final_val, verbose=0)
if len(test_y) > 0:
    predict_test = model.predict_classes(final_test, verbose=0)
"""
