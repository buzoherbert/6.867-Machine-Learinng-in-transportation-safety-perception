#Python 3 file

import sys
from scipy import stats
from random import randrange
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, PReLU, LeakyReLU #, Flatten
from keras.regularizers import l2
from keras import initializers
from sklearn import preprocessing
import math
import matplotlib.pyplot as plt
from process_data import Data
from sklearn.metrics import f1_score

TEST_RATIO = 0.2 #float from 0.0 to 1.0
VAL_RATIO = 0.2 #float from 0.0 to 1.0
NORMALIZE = True #normalize data in "total_passenger_count", "total_female_count", "empty_seats", "haversine"
BATCH_SIZE = 5
INCLUDE = sys.argv[1] #sys.argv[1] #one of "trip_var", "instant_var", "perception_var", "contextual_var", "sociodemographic_var", "ALL"
ACTIVATION = "relu"
FILENAMES = ['final_data_0.csv','final_data_1.csv','final_data_2.csv','final_data_3.csv','final_data_4.csv']
F1METHOD = 'macro'
NUM_EPOCHS = 50
M1 = 10
M2 = None #Set None to remove second layer
Dropout_rates = [0.0, 0.1, 0.2, 0.3] #[0.0, 0.1, 0.2, 0.3] # set 0.0 to disable dropout
Regularizations = [0, 0.001, 0.01, 0.1] #[0, 0.001, 0.01, 0.1]
#DDOF = 0

print("M1: ", M1)
print("M2: ", M2)



### BEST HYPERPARAMS:
"""
for M1 = 10, M2 = 10: Dropout = 0, Reg = 0.01
final val rates (mean, stderr):  0.377173913043 ,  0.0300833750067
final val f1 (mean, stderr):  0.306268324648 ,  0.0256436533925

for M1 = 5, M2 = 5: Dropout = 0, Reg = 0
final val rates (mean, stderr):  0.360869565217 ,  0.0244504823461
final val f1 (mean, stderr):  0.291170404322 ,  0.0213765783738

for M1 = 10, M2 = None: Dropout = 0, Reg = 0.001
final val rates (mean, stderr):  0.352173913043 ,  0.0215754709144
final val f1 (mean, stderr):  0.305646153197 ,  0.0154163018304
"""


SHOW_CONFUSIONS = True
SHOWPLOTS = False

#set COMPARE to true if you want to compare our model with a "mode model" (predict everything to be the most probable perception value)
# and a "random model" (predict perception value x with probability weighted according to the distribution of perception x occuring)
COMPARE = False

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
        model.add(Dense(M1, activation=ACTIVATION, input_dim=input_dim, kernel_regularizer=l2(regularization)))
        if dropout_rate > 0:
            model.add(Dropout(dropout_rate))
        if M2 != None:
            model.add(Dense(M2, activation=ACTIVATION, kernel_regularizer=l2(regularization)))
            if dropout_rate > 0:
                model.add(Dropout(dropout_rate))
        model.add(Dense(5))
        model.add(Activation('softmax'))

        model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
        self.model = model

    def train(self, train_x, train_y, val_x, val_y, test_x, test_y, num_epochs):
        trainclass_y = keras.utils.to_categorical(train_y)
        
        train_rates = []
        val_rates = []
        test_rates = []

        train_f1 = []
        val_f1 = []
        test_f1 = []

        model = self.model
        if SHOWPLOTS:
            for i in range(num_epochs):
                model.fit(train_x, trainclass_y, epochs=1, batch_size=BATCH_SIZE, verbose=0)
                predict_train = model.predict_classes(train_x, verbose=0)
                train_rates.append(get_acc_rate(train_y, predict_train))
                train_f1.append(f1_score(train_y, predict_train, average=F1METHOD))
                predict_val = model.predict_classes(val_x, verbose=0)
                val_rates.append(get_acc_rate(val_y, predict_val))
                val_f1.append(f1_score(val_y, predict_val, average=F1METHOD))
                if len(test_y) > 0:
                    predict_test = model.predict_classes(test_x, verbose=0)
                    test_rates.append(get_acc_rate(test_y, predict_test))
                    test_f1.append(f1_score(test_y, predict_test, average=F1METHOD))
        else:
            model.fit(train_x, trainclass_y, epochs=num_epochs, batch_size=BATCH_SIZE, verbose=0)
            predict_train = model.predict_classes(train_x, verbose=0)
            train_rates.append(get_acc_rate(train_y, predict_train))
            train_f1.append(f1_score(train_y, predict_train, average=F1METHOD))
            predict_val = model.predict_classes(val_x, verbose=0)
            val_rates.append(get_acc_rate(val_y, predict_val))
            val_f1.append(f1_score(val_y, predict_val, average=F1METHOD))
            if len(test_y) > 0:
                predict_test = model.predict_classes(test_x, verbose=0)
                test_rates.append(get_acc_rate(test_y, predict_test))
                test_f1.append(f1_score(test_y, predict_test, average=F1METHOD))

        return train_rates, val_rates, test_rates, train_f1, val_f1, test_f1, predict_train, predict_val, predict_test, model

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
        total += (Y_true[i][0] - Y_pred[i])**2
    return total

def osr2(Y_true, Y_pred, train_y):
    sse_val = sse(Y_true, Y_pred)
    pred_mean = np.mean(train_y)
    sst = sse(Y_true, np.array([pred_mean for i in range(Y_true.size)]))
    osr2 = 1.0 - sse_val/sst
    return osr2

def get_avg_sse(y_trues, y_preds):
    sses = []
    for i in range(len(y_trues)):
        sses.append(sse(y_trues[i], y_preds[i]))
    return np.mean(sses), stats.sem(sses)

def get_avg_osr2(y_trues, y_preds, train_ys):
    osr2s = []
    for i in range(len(y_trues)):
        osr2s.append(osr2(y_trues[i], y_preds[i], train_ys[i]))
    return np.mean(osr2s), stats.sem(osr2s)


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

accs_all = []
f1s_all = []

for dr in Dropout_rates:
    for regul in Regularizations:
        train_rates, val_rates, test_rates, train_f1, val_f1, test_f1, predict_train, predict_val, predict_test, models = [], [], [], [], [], [], [], [], [], []
        for i in range(len(FILENAMES)):
            input_dim = all_train_x[i].shape[1]
            model_class = Model(M1, M2, dr, input_dim, regul)
            data = model_class.train(all_train_x[i], all_train_y[i], all_val_x[i], all_val_y[i], all_test_x[i], all_test_y[i], NUM_EPOCHS)
            train_rates.append(np.array(data[0]))
            val_rates.append(np.array(data[1]))
            test_rates.append(np.array(data[2]))
            train_f1.append(np.array(data[3]))
            val_f1.append(np.array(data[4]))
            test_f1.append(np.array(data[5]))
            predict_train.append(data[6])
            predict_val.append(data[7])
            predict_test.append(data[8])
            models.append(data[9])

        final_train_rates = np.stack(train_rates)
        final_val_rates = np.stack(val_rates)
        final_test_rates = np.stack(test_rates)
        final_train_f1 = np.stack(train_f1)
        final_val_f1 = np.stack(val_f1)
        final_test_f1 = np.stack(test_f1)

        print("Dropout ", dr, ", regul ", regul, ":")
        print("final val sses (mean, stderr): ", get_avg_sse(all_val_y, predict_val))
        print("final val osr2 (mean, stderr): ", get_avg_osr2(all_val_y, predict_val, all_train_y))
        print("final val acc (mean, stderr): ", np.mean(final_val_rates[:, -1]), ", ", stats.sem(final_val_rates[:, -1]))
        print("final val f1 (mean, stderr): ", np.mean(final_val_f1[:, -1]), ", ", stats.sem(final_val_f1[:, -1]))

        print("final test sses (mean, stderr): ", get_avg_sse(all_test_y, predict_test))
        print("final test osr2 (mean, stderr): ", get_avg_osr2(all_test_y, predict_test, all_train_y))        
        print("final test acc (mean, stderr): ", np.mean(final_test_rates[:, -1]), ", ", stats.sem(final_test_rates[:, -1]))
        print("final test f1 (mean, stderr): ", np.mean(final_test_f1[:, -1]), ", ", stats.sem(final_test_f1[:, -1]))
        
        print("final train acc (mean, stderr): ", np.mean(final_train_rates[:, -1]), ", ", stats.sem(final_train_rates[:, -1]))
        print("final train f1 (mean, stderr): ", np.mean(final_train_f1[:, -1]), ", ", stats.sem(final_train_f1[:, -1]))

        accs_all.append((dr, regul, np.mean(final_val_rates[:, -1])))
        f1s_all.append((dr, regul, np.mean(final_val_f1[:, -1])))

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
            #plt.title("average accuracy vs epoch number, using dropout {}, lambda {}".format(dr, regul))
            plt.title("average accuracy vs epoch number, no dropout and regularization")
            plt.legend()
            plt.show()

            plt.plot(np.mean(final_train_f1, axis=0), 'go-', label="train")
            plt.plot(np.mean(final_val_f1, axis=0), 'ro-', label="val")
            plt.plot(np.mean(final_test_f1, axis=0), 'bo-', label="test")
            #plt.title("average f1 score vs epoch number, using dropout {}, lambda {}".format(dr, regul))
            plt.title("average f1 score vs epoch number, no dropout and regularization")
            plt.legend()
            plt.show()

print("ACCS ALL")
print(accs_all)
print("F1s ALL")
print(f1s_all)

def weighted_random(weights):
    x = randrange(sum(weights))
    for i in range(len(weights)):
        if x < weights[i]:
            return i
        else:
            x = x - weights[i]

if COMPARE:
    acc_means = []
    acc_randoms = []
    f1_means = []
    f1_randoms = []
    sse_means = []
    sse_randoms = []
    osr2_means = []
    osr2_randoms = []

    mean_preds = []
    random_preds = []
    mean_preds_int = []
    
    for i in range(len(FILENAMES)):
        train_y = all_train_y[i]
        val_y = all_val_y[i]
        test_y = all_test_y[i]
        
        trainclass_y = keras.utils.to_categorical(train_y)
        sums = np.sum(trainclass_y, axis=0)
        #mode = max(range(5), key=lambda x: sums[x])
        #print(mode)
        #mode_prediction = np.array([mode for i in range(test_y.size)])

        mean = np.sum(train_y)/train_y.size
        mean_prediction = np.array([mean for i in range(test_y.size)])
        mean_prediction_int = np.clip(mean_prediction.round().astype(int), 0, 4)

        random_pred = np.array([weighted_random(sums) for i in range(test_y.size)])
        sse_means.append(sse(test_y, mean_prediction))
        sse_randoms.append(sse(test_y, random_pred))
        osr2_means.append(osr2(test_y, mean_prediction, train_y))
        osr2_randoms.append(osr2(test_y, random_pred, train_y))
        acc_means.append(get_acc_rate(test_y, mean_prediction_int))
        acc_randoms.append(get_acc_rate(test_y, random_pred))
        f1_means.append(f1_score(test_y, mean_prediction_int, average=F1METHOD))
        f1_randoms.append(f1_score(test_y, random_pred, average=F1METHOD))

        random_preds.append(random_pred)
        mean_preds.append(mean_prediction)
        mean_preds_int.append(mean_prediction_int)

    print("sse using mean: ", (np.mean(sse_means), stats.sem(sse_means)))
    print("osr2 using mean: ", (np.mean(osr2_means), stats.sem(osr2_means)))
    print("accuracy rate using mean: ", (np.mean(acc_means), stats.sem(acc_means)))
    print("f1 rate using mean: ", (np.mean(f1_means), stats.sem(f1_means)))

    print("sse using random: ", (np.mean(sse_randoms), stats.sem(sse_randoms)))
    print("osr2 using random: ", (np.mean(osr2_randoms), stats.sem(osr2_randoms)))
    print("accuracy rate using random: ", (np.mean(acc_randoms), stats.sem(acc_randoms)))
    print("f1 rate using random: ", (np.mean(f1_randoms), stats.sem(f1_randoms)))

    print("Confusion test matrix for mode:")
    print(get_total_mtx(all_test_y, mean_preds_int))
    
    print("Confusion test matrix for random:")
    print(get_total_mtx(all_test_y, random_preds))

"""
sse using mean:  (326.65691746733398, 9.9850555676649755)
osr2 using mean:  (0.0, 0.0)
accuracy rate using mean:  (0.30162162162162159, 0.011517446352790556)
f1 rate using mean:  (0.092594372632810312, 0.0027313774983301624)
sse using random:  (677.0, 29.022405138099767)
osr2 using random:  (-1.0749901726911912, 0.07877335455735561)
accuracy rate using random:  (0.22162162162162158, 0.0056692370171359599)
f1 rate using random:  (0.18277240551802154, 0.0083820671926260505)
Confusion test matrix for mode:
[[ 0.          0.          0.23243243  0.          0.        ]
 [ 0.          0.          0.09945946  0.          0.        ]
 [ 0.          0.          0.30162162  0.          0.        ]
 [ 0.          0.          0.24432432  0.          0.        ]
 [ 0.          0.          0.12216216  0.          0.        ]]
Confusion test matrix for random:
[[ 0.04756757  0.02810811  0.07567568  0.04864865  0.03243243]
 [ 0.02162162  0.00972973  0.02810811  0.0227027   0.0172973 ]
 [ 0.07675676  0.03567568  0.10702703  0.05405405  0.02810811]
 [ 0.05837838  0.02378378  0.08324324  0.04756757  0.03135135]
 [ 0.03351351  0.01405405  0.03459459  0.03027027  0.00972973]]
"""
