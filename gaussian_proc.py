#Python 3 file

from scipy import stats
import numpy as np
import math
import matplotlib.pyplot as plt
from process_data import Data
from sklearn.metrics import f1_score

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C


TEST_RATIO = 0.2 #float from 0.0 to 1.0
VAL_RATIO = 0.2 #float from 0.0 to 1.0
NORMALIZE = True #normalize data in "total_passenger_count", "total_female_count", "empty_seats", "haversine"
FILENAMES = ['final_data_4.csv']#,'final_data_1.csv','final_data_2.csv','final_data_3.csv','final_data_4.csv'] #['final_data_4.csv'] #['final_data_0.csv','final_data_1.csv','final_data_2.csv','final_data_3.csv','final_data_4.csv']
MULTS = [5] #[1.0, 2.0, 5.0, 10.0, 20.0] #8.0
LENGTHS = [5] #[1.0, 2.0, 5.0, 10.0, 20.0] #8.0
ALPHA = [1.5] #[1, 1.5, 2, 2.5, 3]
F1METHOD = 'macro'
DDOF = 0

"""
BEST HYPERPARAMS:
ALPHA  0.5 , LENGTH  4.0 :, mult = 1
final val rates (mean, stderr):  268.820392085 ,  21.4774895447
final val osr2 (mean, stderr):  0.098601024214 ,  0.0256095710542

MULT  8.0 , LENGTH  8.0 :
final val rates (mean, stderr):  268.267033012 ,  22.3762852161
final val osr2 (mean, stderr):  0.100676193254 ,  0.0268295600165



MULT  10.0 , LENGTH  10.0 :
final val rates (mean, stderr):  268.302270726 ,  23.1874750669
final val osr2 (mean, stderr):  0.100813698508 ,  0.0261935793535


ALPHA:  1.5
0.415**2 * RBF(length_scale=2.18)
0.475**2 * RBF(length_scale=1.76)
0.43**2 * RBF(length_scale=1.55)
0.474**2 * RBF(length_scale=1.23)
0.341**2 * RBF(length_scale=1.37)
MULT  5 , LENGTH  5 :
final val rates (mean, stderr):  271.672555638 ,  11.6474111202
final val osr2 (mean, stderr):  0.094511697369 ,  0.0144118428945
final val accs (mean, stderr):  (0.28972972972972971, 0.0083039413490471475)
final val f1s (mean, stderr):  (0.13049795612455845, 0.0086136820509058257)
final test rates (mean, stderr):  302.859849122 ,  6.46010233317
final test osr2 (mean, stderr):  0.07153083879 ,  0.0134602442436
final test accs (mean, stderr):  (0.25513513513513508, 0.012607463556422265)
final test f1s (mean, stderr):  (0.12469288695156085, 0.014113865438063679)
final train rates (mean, stderr):  808.716161779 ,  9.25338462679
final train osr2 (mean, stderr):  0.123557461014 ,  0.0129253794363
Confusion matrix for training data:
[[ 0.          0.08239564  0.14555354  0.00181488  0.        ]
 [ 0.          0.02250454  0.09038113  0.00181488  0.        ]
 [ 0.          0.0646098   0.25263158  0.0076225   0.        ]
 [ 0.          0.02032668  0.18802178  0.0123412   0.        ]
 [ 0.          0.00725953  0.09437387  0.00834846  0.        ]]
Confusion matrix for validation data:
[[ 0.          0.06162162  0.14594595  0.00540541  0.        ]
 [ 0.          0.02486486  0.11243243  0.          0.        ]
 [ 0.          0.06054054  0.25945946  0.00540541  0.        ]
 [ 0.          0.01297297  0.19891892  0.00540541  0.        ]
 [ 0.          0.00972973  0.09297297  0.00432432  0.        ]]
Confusion matrix for test data:
[[ 0.          0.06810811  0.16        0.00432432  0.        ]
 [ 0.          0.02702703  0.07027027  0.00216216  0.        ]
 [ 0.          0.07351351  0.22054054  0.00756757  0.        ]
 [ 0.          0.02810811  0.20864865  0.00756757  0.        ]
 [ 0.          0.00756757  0.10918919  0.00540541  0.        ]]
"""

SHOW_CONFUSIONS = True
SHOWPLOTS = True

all_train_x, all_train_y, all_val_x, all_val_y, all_test_x, all_test_y = [], [], [], [], [], []

for filename in FILENAMES:
    data_loader = Data(VAL_RATIO, TEST_RATIO, "ALL", filename, normalize = NORMALIZE)

    title_x, title_y = data_loader.get_title()
    train_x, train_y = data_loader.get_train_data()
    val_x, val_y = data_loader.get_val_data()
    test_x, test_y = data_loader.get_test_data()

    if title_x[-1] == "latitude":
        lat_train = train_x[:, -1]
        lon_train = train_x[:, -2]
        lat_val = val_x[:, -1]
        lon_val = val_x[:, -2]
        lat_test = test_x[:, -1]
        lon_test = test_x[:, -2]
    elif title_x[-1] == "longitude":
        lat_train = train_x[:, -2]
        lon_train = train_x[:, -1]
        lat_val = val_x[:, -2]
        lon_val = val_x[:, -1]
        lat_test = test_x[:, -2]
        lon_test = test_x[:, -1]

    center_vert = np.mean(lat_train)
    center_hor = np.mean(lon_train)

    Lat_to_mile = 69 # 1 degree lat ~= 69 miles
    Lon_to_mile = math.cos(center_vert*math.pi/180) * 69.17 # ratio of 1 degree lon to mile

    vert_train = np.clip((lat_train.reshape((-1, 1)) - center_vert)*Lat_to_mile, -5, None)
    hor_train = (lon_train.reshape((-1, 1)) - center_hor)*Lon_to_mile
    vert_val = np.clip((lat_val.reshape((-1, 1)) - center_vert)*Lat_to_mile, -5, None)
    hor_val = (lon_val.reshape((-1, 1)) - center_hor)*Lon_to_mile
    vert_test = np.clip((lat_test.reshape((-1, 1)) - center_vert)*Lat_to_mile, -5, None)
    hor_test = (lon_test.reshape((-1, 1)) - center_hor)*Lon_to_mile

    train_x = np.concatenate((hor_train, vert_train), axis=1)
    val_x = np.concatenate((hor_val, vert_val), axis=1)
    test_x = np.concatenate((hor_test, vert_test), axis=1)

    all_train_x.append(train_x)
    all_train_y.append(train_y)
    all_val_x.append(val_x)
    all_val_y.append(val_y)
    all_test_x.append(test_x)
    all_test_y.append(test_y)


import pdb
from numpy import *
import pylab as pl

# X is data matrix (each row is a data point)
# Y is desired output (1 or -1)
# scoreFn is a function of a data point
# values is a list of values to plot

def plotDecisionBoundary(X, Y, scoreFn, values, title = ""):
    # Plot the decision boundary. For that, we will asign a score to
    # each point in the mesh [x_min, m_max]x[y_min, y_max].
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    h = max((x_max-x_min)/200., (y_max-y_min)/200.)
    xx, yy = meshgrid(arange(x_min, x_max, h),
                      arange(y_min, y_max, h))
    zz = array([scoreFn(x) for x in c_[xx.ravel(), yy.ravel()]])
    zz = zz.reshape(xx.shape)
    pl.figure()
    CS = pl.contour(xx, yy, zz, values, colors = 'green', linestyles = 'solid', linewidths = 2)
    pl.clabel(CS, fontsize=9, inline=1)
    # Plot the training points
    pl.scatter(X[:, 0], X[:, 1], c=Y[:, 0]+1, s=50, cmap = pl.cm.cool_r)
    pl.colorbar()
    pl.title(title)
    pl.axis('tight')

def get_acc_rate(Y_true, Y_pred):
    n = Y_true.size
    err = 0
    for i in range(n):
        if Y_true[i][0] != Y_pred[i]:
            err += 1
    return 1.0 - float(err)/n

def get_avg_acc(y_trues, y_preds):
    acc = []
    for i in range(len(y_trues)):
        acc.append(get_acc_rate(y_trues[i], y_preds[i]))
    return np.mean(acc), stats.sem(acc)

def get_avg_f1(y_trues, y_preds):
    f1s = []
    for i in range(len(y_trues)):
        f1s.append(f1_score(y_trues[i], y_preds[i], average=F1METHOD))
    return np.mean(f1s), stats.sem(f1s)

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

for m in MULTS:
    for l in LENGTHS:
        for a in ALPHA:
            print("ALPHA: ", a)
            train_rates, val_rates, test_rates, train_osr2, val_osr2, test_osr2, predict_train, predict_val, predict_test, models = [], [], [], [], [], [], [], [], [], []
            for i in range(len(FILENAMES)):
                all_train_x[i], all_train_y[i], all_val_x[i], all_val_y[i], all_test_x[i], all_test_y[i]

                # Instanciate a Gaussian Process model
                #kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
                #gp = GaussianProcessRegressor(kernel=kernel, alpha=1.0, n_restarts_optimizer=10)
                kernel = m**2 * RBF(l)
                gp = GaussianProcessRegressor(kernel=kernel, alpha=a, normalize_y = True, n_restarts_optimizer=5)#, optimizer=None)

                gp.fit(all_train_x[i], all_train_y[i])
                print(gp.kernel_)

                train_pred = gp.predict(all_train_x[i])
                val_pred = gp.predict(all_val_x[i])
                test_pred = gp.predict(all_test_x[i])

                sse_train = sse(all_train_y[i], train_pred)
                osr2_train = osr2(all_train_y[i], train_pred, all_train_y[i])
                sse_val = sse(all_val_y[i], val_pred)
                osr2_val = osr2(all_val_y[i], val_pred, all_train_y[i])
                sse_test = sse(all_test_y[i], test_pred)
                osr2_test = osr2(all_test_y[i], test_pred, all_train_y[i])

                predict_train_int = np.clip(train_pred.round().astype(int), 0, 4)
                predict_val_int = np.clip(val_pred.round().astype(int), 0, 4)
                predict_test_int = np.clip(test_pred.round().astype(int), 0, 4)

                train_rates.append(sse_train)
                val_rates.append(sse_val)
                test_rates.append(sse_test)
                train_osr2.append(osr2_train)
                val_osr2.append(osr2_val)
                test_osr2.append(osr2_test)
                models.append(gp)

                predict_train.append(predict_train_int)
                predict_val.append(predict_val_int)
                predict_test.append(predict_test_int)

                if SHOWPLOTS:

                    def predictScore(x):
                        y_pred = gp.predict(x.reshape(1, -1)) + 1
                        return y_pred
                    title = "safety perception vs location (alpha=" + str(a) + ")"
                    

                    plotDecisionBoundary(all_test_x[i], all_test_y[i], predictScore, [2.25, 2.5, 2.75, 3.0, 3.25, 3.5], title = "test set " + title)
                    #plotDecisionBoundary(all_val_x[i], all_val_y[i], predictScore, [0.0, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0], title = "val")
                    plotDecisionBoundary(all_train_x[i], all_train_y[i], predictScore, [2.25, 2.5, 2.75, 3.0, 3.25, 3.5], title = "training set " + title)


            print("MULT ", m, ", LENGTH ", l, ":")
            print("final val rates (mean, stderr): ", np.mean(val_rates), ", ", stats.sem(val_rates))
            print("final val osr2 (mean, stderr): ", np.mean(val_osr2), ", ", stats.sem(val_osr2))
            print("final val accs (mean, stderr): ", get_avg_acc(all_val_y, predict_val))
            print("final val f1s (mean, stderr): ", get_avg_f1(all_val_y, predict_val))

            print("final test rates (mean, stderr): ", np.mean(test_rates), ", ", stats.sem(test_rates))
            print("final test osr2 (mean, stderr): ", np.mean(test_osr2), ", ", stats.sem(test_osr2))
            print("final test accs (mean, stderr): ", get_avg_acc(all_test_y, predict_test))
            print("final test f1s (mean, stderr): ", get_avg_f1(all_test_y, predict_test))

            print("final train rates (mean, stderr): ", np.mean(train_rates), ", ", stats.sem(train_rates))
            print("final train osr2 (mean, stderr): ", np.mean(train_osr2), ", ", stats.sem(train_osr2))

            if SHOWPLOTS:
                pl.show()

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

            
