
import keras
from sklearn.linear_model import LogisticRegression
from scipy import stats
from sklearn.metrics import f1_score
from process_data import Data
import numpy as np

TEST_RATIO = 0.2 #float from 0.0 to 1.0
VAL_RATIO = 0.2 #float from 0.0 to 1.0
NORMALIZE = True #normalize data in "total_passenger_count", "total_female_count", "empty_seats", "haversine"
INCLUDE = "ALL" #sys.argv[1] #one of "trip_var", "instant_var", "perception_var", "contextual_var", "sociodemographic_var", "ALL"
ACTIVATION = "relu"
FILENAMES = ['final_data_0.csv','final_data_1.csv','final_data_2.csv','final_data_3.csv','final_data_4.csv']
F1METHOD = 'macro'
regularization = [0.001, 0.01, 0.1, 1, 10, 100, 1000]

SHOW_CONFUSIONS = True


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
    def __init__(self, regul):
        self.regul = regul

    def train(self, train_x, train_y, val_x, val_y, test_x, test_y):
        trainclass_y = keras.utils.to_categorical(train_y)
        
        train_rates = []
        val_rates = []
        test_rates = []

        train_f1 = []
        val_f1 = []
        test_f1 = []

        model = LogisticRegression(solver='saga', multi_class='multinomial', C = self.regul)
        
        model.fit(train_x, train_y)
        predict_train = model.predict(train_x)
        train_rates.append(get_acc_rate(train_y, predict_train))
        train_f1.append(f1_score(train_y, predict_train, average=F1METHOD))
        predict_val = model.predict(val_x)
        val_rates.append(get_acc_rate(val_y, predict_val))
        val_f1.append(f1_score(val_y, predict_val, average=F1METHOD))
        if len(test_y) > 0:
            predict_test = model.predict(test_x)
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


for regul in regularization:
    train_rates, val_rates, test_rates, train_f1, val_f1, test_f1, predict_train, predict_val, predict_test, models = [], [], [], [], [], [], [], [], [], []
    for i in range(len(FILENAMES)):
        input_dim = all_train_x[i].shape[1]
        model_class = Model(regul)
        data = model_class.train(all_train_x[i], all_train_y[i], all_val_x[i], all_val_y[i], all_test_x[i], all_test_y[i])
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

    print("C: ", regul)
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

    accs_all.append((regul, np.mean(final_val_rates[:, -1])))
    f1s_all.append((regul, np.mean(final_val_f1[:, -1])))

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

print("ACCS ALL")
print(accs_all)
print("F1s ALL")
print(f1s_all)





    
