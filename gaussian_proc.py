#Python 3 file

import numpy as np
import math
import matplotlib.pyplot as plt
from process_data import Data
from sklearn.metrics import f1_score

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C


TEST_RATIO = 0.0 #float from 0.0 to 1.0
VAL_RATIO = 0.3 #float from 0.0 to 1.0
NORMALIZE = True #normalize data in "total_passenger_count", "total_female_count", "empty_seats", "haversine"
INCLUDE = "ALL" #one of "trip_var", "perception_var", "contextual_var", "sociodemographic_var", "ALL"
FILENAME = 'final_data_2.csv'

data_loader = Data(VAL_RATIO, TEST_RATIO, INCLUDE, FILENAME, normalize = NORMALIZE)

title_x, title_y = data_loader.get_title()
train_x, train_y = data_loader.get_train_data()
val_x, val_y = data_loader.get_val_data()
test_x, test_y = data_loader.get_test_data()

if title_x[-1] == "latitude":
    lat_train = train_x[:, -1]
    lon_train = train_x[:, -2]
    lat_val = val_x[:, -1]
    lon_val = val_x[:, -2]
elif title_x[-1] == "longitude":
    lat_train = train_x[:, -2]
    lon_train = train_x[:, -1]
    lat_val = val_x[:, -2]
    lon_val = val_x[:, -1]

center_vert = np.mean(lat_train)
center_hor = np.mean(lon_train)

Lat_to_mile = 69 # 1 degree lat ~= 69 miles
Lon_to_mile = math.cos(center_vert*math.pi/180) * 69.17 # ratio of 1 degree lon to mile

vert_train = np.clip((lat_train.reshape((-1, 1)) - center_vert)*Lat_to_mile, -5, None)
hor_train = (lon_train.reshape((-1, 1)) - center_hor)*Lon_to_mile
vert_val = np.clip((lat_val.reshape((-1, 1)) - center_vert)*Lat_to_mile, -5, None)
hor_val = (lon_val.reshape((-1, 1)) - center_hor)*Lon_to_mile

train_x = np.concatenate((hor_train, vert_train), axis=1)
val_x = np.concatenate((hor_val, vert_val), axis=1)


# Instanciate a Gaussian Process model
kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
#gp = GaussianProcessRegressor(kernel=kernel, alpha=1.0, n_restarts_optimizer=10)
gp = GaussianProcessRegressor(kernel=None, alpha=1.0, n_restarts_optimizer=10, normalize_y = True)

gp.fit(train_x, train_y)

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
    pl.scatter(X[:, 0], X[:, 1], c=Y[:, 0], s=50, cmap = pl.cm.cool)
    pl.colorbar()
    pl.title(title)
    pl.axis('tight')

def predictScore(x):
    y_pred = gp.predict(x.reshape(1, -1))
    return y_pred

plotDecisionBoundary(val_x, val_y, predictScore, [0.0, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0], title = "val")
plotDecisionBoundary(train_x, train_y, predictScore, [0.0, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0], title = "train")
pl.show()

def sse(Y_true, Y_pred):
    n = Y_true.size
    total = 0
    for i in range(n):
        total += (Y_true[i][0] - Y_pred[i][0])**2
    return total

y_pred = gp.predict(val_x)
sse_final = sse(val_y, y_pred)
print("final sse for val data: ", sse_final)

pred_mean = np.mean(train_y)
sst = sse(val_y, np.array([[pred_mean] for i in range(val_y.size)]))

print("sst for val data: ", sst)

osr2 = 1.0 - sse_final/sst

print("osr2: ", osr2)
