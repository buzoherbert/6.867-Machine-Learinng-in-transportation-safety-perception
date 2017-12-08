#Python 3 file

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
INCLUDE = "ALL" #one of "trip_var", "perception_var", "contextual_var", "sociodemographic_var", "ALL"
ACTIVATION = "relu"
FILENAMES = ['final_data_0.csv','final_data_1.csv','final_data_2.csv','final_data_3.csv','final_data_4.csv']
F1METHOD = 'macro'
NUM_EPOCHS = 50
M1 = 10
M2 = 10 #Set None to remove second layer
Dropout_rates = [] #[0.0, 0.1, 0.2, 0.3] # set 0.0 to disable dropout
Regularizations = [] #[0, 0.001, 0.01, 0.1]
DDOF = 0

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


SHOW_CONFUSIONS = False
SHOWPLOTS = False

#set COMPARE to true if you want to compare our model with a "mode model" (predict everything to be the most probable perception value)
# and a "random model" (predict perception value x with probability weighted according to the distribution of perception x occuring)
COMPARE = True

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

        return train_rates, val_rates, test_rates, train_f1, val_f1, test_f1, predict_train, predict_val, predict_test, model

def get_acc_rate(Y_true, Y_pred):
    n = Y_true.size
    err = 0
    for i in range(n):
        if Y_true[i][0] != Y_pred[i]:
            err += 1
    return 1.0 - float(err)/n

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
        print("final val rates (mean, stderr): ", np.mean(final_val_rates[:, -1]), ", ", np.std(final_val_rates[:, -1], ddof=DDOF))
        print("final val f1 (mean, stderr): ", np.mean(final_val_f1[:, -1]), ", ", np.std(final_val_f1[:, -1], ddof=DDOF))
        
        #print("final test rates (mean, stderr): ", np.mean(final_test_rates[:, -1]), ", ", np.std(final_test_rates[:, -1], ddof=DDOF))
        #print("final test f1 (mean, stderr): ", np.mean(final_test_f1[:, -1]), ", ", np.std(final_test_f1[:, -1], ddof=DDOF))
        
        #print("final train rates (mean, stderr): ", np.mean(final_train_rates[:, -1]), ", ", np.std(final_train_rates[:, -1], ddof=DDOF))
        #print("final train f1 (mean, stderr): ", np.mean(final_train_f1[:, -1]), ", ", np.std(final_train_f1[:, -1], ddof=DDOF))

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
            plt.title("average accuracy rate vs epoch number")
            plt.legend()
            plt.show()

            plt.plot(np.mean(final_train_f1, axis=0), 'go-', label="train")
            plt.plot(np.mean(final_val_f1, axis=0), 'ro-', label="val")
            plt.plot(np.mean(final_test_f1, axis=0), 'bo-', label="test")
            plt.title("average f1 score vs epoch number")
            plt.legend()
            plt.show()


def weighted_random(weights):
    x = randrange(sum(weights))
    for i in range(len(weights)):
        if x < weights[i]:
            return i
        else:
            x = x - weights[i]

if COMPARE:
    acc_modes = []
    acc_randoms = []
    f1_modes = []
    f1_randoms = []

    mode_preds = []
    random_preds = []
    
    for i in range(len(FILENAMES)):
        train_y = all_train_y[i]
        val_y = all_val_y[i]
        test_y = all_test_y[i]
        
        trainclass_y = keras.utils.to_categorical(train_y)
        sums = np.sum(trainclass_y, axis=0)
        mode = max(range(5), key=lambda x: sums[x])
        mode_prediction = np.array([mode for i in range(test_y.size)])
        random_pred = np.array([weighted_random(sums) for i in range(test_y.size)])
        acc_modes.append(get_acc_rate(test_y, mode_prediction))
        acc_randoms.append(get_acc_rate(test_y, random_pred))
        f1_modes.append(f1_score(test_y, mode_prediction, average=F1METHOD))
        f1_randoms.append(f1_score(test_y, random_pred, average=F1METHOD))

        random_preds.append(random_pred)
        mode_preds.append(mode_prediction)

    print("accuracy rate using mode: ", np.mean(acc_modes))
    print("f1 rate using mode: ", np.mean(f1_modes))

    print("error rate using random: ", np.mean(acc_randoms))
    print("f1 rate using random: ", np.mean(f1_randoms))

    print("Confusion test matrix for mode:")
    print(get_total_mtx(all_test_y, mode_preds))
    
    print("Confusion test matrix for random:")
    print(get_total_mtx(all_test_y, random_preds))
