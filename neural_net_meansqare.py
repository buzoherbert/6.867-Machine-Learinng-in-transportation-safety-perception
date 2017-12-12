#Python 3 file

from scipy import stats
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
INCLUDE = "sociodemographic_var" #one of "trip_var", "instant_var", "perception_var", "contextual_var", "sociodemographic_var", "ALL"
FILENAMES = ['final_data_0.csv','final_data_1.csv','final_data_2.csv','final_data_3.csv','final_data_4.csv']
F1METHOD = 'macro'
NUM_EPOCHS = 30
M1 = 10
M2 = 10 #Set None to remove second layer
Dropout_rates = [0.0, 0.1, 0.2, 0.3] #[0.0, 0.1, 0.2, 0.3] # set 0.0 to disable dropout
Regularizations = [0, 0.01, 0.1] #[0, 0.001, 0.01, 0.1]
#DDOF = 1

print("INCLUDING: ", INCLUDE)

if M2 == None:
    print("M1")
else:
    print("M2")

SHOW_CONFUSIONS = True
SHOWPLOTS = False

"""
INCLUDING: ALL
M1
Dropout  0.2 , regul  0.1 :
final val rates (mean, stderr):  256.505135675 ,  8.77489537065
final val osr2 (mean, stderr):  0.143299984119 ,  0.0200284876675
final val accs (mean, stderr):  (0.25729729729729722, 0.010756620941693187)
final val f1s (mean, stderr):  (0.15122205778571673, 0.0035503890816936017)
final test rates (mean, stderr):  279.438917161 ,  6.8142550109
final test osr2 (mean, stderr):  0.142382407772 ,  0.0259502369858
final test accs (mean, stderr):  (0.22702702702702701, 0.016660578386402648)
final test f1s (mean, stderr):  (0.1346136300656329, 0.0099041718341176116)
final train rates (mean, stderr):  613.855579604 ,  15.3963678263
final train osr2 (mean, stderr):  0.334803396743 ,  0.0170529480449
Confusion matrix for training data:
[[ 0.00362976  0.16261343  0.06025408  0.00326679  0.        ]
 [ 0.00108893  0.04355717  0.06860254  0.00145191  0.        ]
 [ 0.          0.08529946  0.21960073  0.0199637   0.        ]
 [ 0.          0.01560799  0.16479129  0.04029038  0.        ]
 [ 0.          0.00653358  0.06678766  0.03629764  0.00036298]]
Confusion matrix for validation data:
[[ 0.          0.12756757  0.07135135  0.01405405  0.        ]
 [ 0.00108108  0.04216216  0.09081081  0.00324324  0.        ]
 [ 0.          0.10378378  0.18918919  0.03243243  0.        ]
 [ 0.          0.03135135  0.16        0.02594595  0.        ]
 [ 0.          0.01297297  0.07027027  0.02378378  0.        ]]
Confusion matrix for test data:
[[ 0.00108108  0.12540541  0.09405405  0.01189189  0.        ]
 [ 0.00108108  0.03243243  0.06162162  0.00432432  0.        ]
 [ 0.          0.11567568  0.17081081  0.01513514  0.        ]
 [ 0.00108108  0.02918919  0.19135135  0.0227027   0.        ]
 [ 0.          0.01297297  0.08216216  0.02702703  0.        ]]

INCLUDING: ALL
M2
Dropout  0.0 , regul  0.1 :
final val rates (mean, stderr):  253.910090047 ,  9.1682679506
final val osr2 (mean, stderr):  0.152022140933 ,  0.0212836708947
final val accs (mean, stderr):  (0.26810810810810815, 0.0097597135596042134)
final val f1s (mean, stderr):  (0.16570962768983238, 0.0049427708697594103)
final test rates (mean, stderr):  275.270314221 ,  2.96525337991
final test osr2 (mean, stderr):  0.154564300741 ,  0.024050901293
final test accs (mean, stderr):  (0.24108108108108112, 0.016537360584625252)
final test f1s (mean, stderr):  (0.14599118952012247, 0.011578303128493873)
final train rates (mean, stderr):  616.911607182 ,  21.8872449347
final train osr2 (mean, stderr):  0.331653590463 ,  0.0226561346379
Confusion matrix for training data:
[[ 0.00508167  0.15353902  0.06678766  0.00435572  0.        ]
 [ 0.00108893  0.03956443  0.07150635  0.00254083  0.        ]
 [ 0.          0.08058076  0.21923775  0.02504537  0.        ]
 [ 0.          0.01524501  0.15462795  0.0508167   0.        ]
 [ 0.          0.00653358  0.06170599  0.04174229  0.        ]]
Confusion matrix for validation data:
[[ 0.00324324  0.12324324  0.07243243  0.01405405  0.        ]
 [ 0.          0.04108108  0.08972973  0.00648649  0.        ]
 [ 0.          0.09945946  0.18918919  0.03675676  0.        ]
 [ 0.          0.02918919  0.15351351  0.03459459  0.        ]
 [ 0.          0.01297297  0.0627027   0.03135135  0.        ]]
Confusion matrix for test data:
[[ 0.00108108  0.12324324  0.09297297  0.01513514  0.        ]
 [ 0.          0.02810811  0.06702703  0.00432432  0.        ]
 [ 0.          0.10810811  0.1772973   0.01621622  0.        ]
 [ 0.00108108  0.03783784  0.17081081  0.03459459  0.        ]
 [ 0.          0.01081081  0.07351351  0.03783784  0.        ]]






INCLUDING:  trip_var
M1
Dropout  0.2 , regul  0.1 :
final val rates (mean, stderr):  285.311667733 ,  11.5544078947
final val osr2 (mean, stderr):  0.0487453611171 ,  0.0121238808187
final val accs (mean, stderr):  (0.28864864864864864, 0.016537360584625248)
final val f1s (mean, stderr):  (0.13725286124726072, 0.010005039644429062)
final test rates (mean, stderr):  316.271119213 ,  9.82615771296
final test osr2 (mean, stderr):  0.0316160187822 ,  0.0100094585737
final test accs (mean, stderr):  (0.23459459459459459, 0.015814852798192204)
final test f1s (mean, stderr):  (0.11071936597603779, 0.0095553735720176978)
final train rates (mean, stderr):  811.559928682 ,  6.2646566466
final train osr2 (mean, stderr):  0.120592963572 ,  0.00783165956712
Confusion matrix for training data:
[[ 0.          0.08203267  0.14736842  0.00036298  0.        ]
 [ 0.          0.02976407  0.08384755  0.00108893  0.        ]
 [ 0.          0.07259528  0.24827586  0.00399274  0.        ]
 [ 0.          0.01705989  0.19854809  0.00508167  0.        ]
 [ 0.          0.01052632  0.09582577  0.00362976  0.        ]]
Confusion matrix for validation data:
[[ 0.          0.06918919  0.14054054  0.00324324  0.        ]
 [ 0.          0.03351351  0.1027027   0.00108108  0.        ]
 [ 0.          0.06702703  0.24864865  0.00972973  0.        ]
 [ 0.          0.01297297  0.19783784  0.00648649  0.        ]
 [ 0.          0.0172973   0.08972973  0.          0.        ]]
Confusion matrix for test data:
[[ 0.          0.07351351  0.15891892  0.          0.        ]
 [ 0.          0.02054054  0.07891892  0.          0.        ]
 [ 0.          0.08864865  0.20972973  0.00324324  0.        ]
 [ 0.          0.02486486  0.21513514  0.00432432  0.        ]
 [ 0.          0.01837838  0.1027027   0.00108108  0.        ]]

INCLUDING:  trip_var
M2
Dropout  0.1 , regul  0.1 :
final val rates (mean, stderr):  283.977971336 ,  11.8243916722
final val osr2 (mean, stderr):  0.0535347588646 ,  0.00944949868596
final val accs (mean, stderr):  (0.2994594594594594, 0.015346021153057708)
final val f1s (mean, stderr):  (0.12596502618893207, 0.0096667387886890943)
final test rates (mean, stderr):  313.217045707 ,  8.51746814319
final test osr2 (mean, stderr):  0.0407016743745 ,  0.00460930883339
final test accs (mean, stderr):  (0.25513513513513508, 0.010312856231534537)
final test f1s (mean, stderr):  (0.11412249777981476, 0.011039228662156715)
final train rates (mean, stderr):  829.571775754 ,  7.89745345435
final train osr2 (mean, stderr):  0.101038920075 ,  0.0102702576227
Confusion matrix for training data:
[[ 0.          0.04827586  0.18076225  0.00072595  0.        ]
 [ 0.          0.01887477  0.09546279  0.00036298  0.        ]
 [ 0.          0.04791289  0.27513612  0.00181488  0.        ]
 [ 0.          0.00725953  0.21016334  0.00326679  0.        ]
 [ 0.          0.00508167  0.1030853   0.00181488  0.        ]]
Confusion matrix for validation data:
[[ 0.          0.04216216  0.16972973  0.00108108  0.        ]
 [ 0.          0.02162162  0.11567568  0.          0.        ]
 [ 0.          0.04324324  0.27675676  0.00540541  0.        ]
 [ 0.          0.00648649  0.20972973  0.00108108  0.        ]
 [ 0.          0.00864865  0.09837838  0.          0.        ]]
Confusion matrix for test data:
[[ 0.          0.03783784  0.19459459  0.          0.        ]
 [ 0.          0.01513514  0.08432432  0.          0.        ]
 [ 0.          0.06486486  0.23459459  0.00216216  0.        ]
 [ 0.          0.01621622  0.2227027   0.00540541  0.        ]
 [ 0.          0.00648649  0.11567568  0.          0.        ]]






INCLUDING:  instant_var
M1
Dropout  0.1 , regul  0.01 :
final val rates (mean, stderr):  287.231978617 ,  10.1304989059
final val osr2 (mean, stderr):  0.0417727265718 ,  0.0115870997281
final val accs (mean, stderr):  (0.27675675675675676, 0.0079074263990960986)
final val f1s (mean, stderr):  (0.1428395920510443, 0.0093968758624633549)
final test rates (mean, stderr):  318.465168165 ,  7.75322712371
final test osr2 (mean, stderr):  0.0224109214361 ,  0.0307446672391
final test accs (mean, stderr):  (0.24756756756756756, 0.020109270527284614)
final test f1s (mean, stderr):  (0.1346157071800243, 0.0136164664151684)
final train rates (mean, stderr):  804.198469001 ,  8.6542525873
final train osr2 (mean, stderr):  0.128494377689 ,  0.0116274865963
Confusion matrix for training data:
[[ 0.          0.09147005  0.13212341  0.0061706   0.        ]
 [ 0.          0.03049002  0.08094374  0.00326679  0.        ]
 [ 0.          0.07586207  0.2323049   0.01669691  0.        ]
 [ 0.          0.02976407  0.17059891  0.02032668  0.        ]
 [ 0.          0.00508167  0.08747731  0.01742287  0.        ]]
Confusion matrix for validation data:
[[ 0.          0.07243243  0.12864865  0.01189189  0.        ]
 [ 0.          0.02486486  0.10594595  0.00648649  0.        ]
 [ 0.          0.07675676  0.23351351  0.01513514  0.        ]
 [ 0.          0.03567568  0.16324324  0.01837838  0.        ]
 [ 0.          0.00972973  0.08216216  0.01513514  0.        ]]
Confusion matrix for test data:
[[ 0.          0.07459459  0.13837838  0.01945946  0.        ]
 [ 0.          0.03027027  0.06486486  0.00432432  0.        ]
 [ 0.          0.08540541  0.20216216  0.01405405  0.        ]
 [ 0.          0.03351351  0.19567568  0.01513514  0.        ]
 [ 0.          0.01297297  0.09621622  0.01297297  0.        ]]

INCLUDING:  instant_var
M2
Dropout  0.3 , regul  0.01 :
final val rates (mean, stderr):  286.144672427 ,  10.2375653708
final val osr2 (mean, stderr):  0.0453927567494 ,  0.0143229355431
final val accs (mean, stderr):  (0.30162162162162159, 0.012254901674329343)
final val f1s (mean, stderr):  (0.1282359002017312, 0.011711568882277111)
final test rates (mean, stderr):  317.04174605 ,  8.77587268885
final test osr2 (mean, stderr):  0.0283732226122 ,  0.0177748302752
final test accs (mean, stderr):  (0.27675675675675671, 0.019065072528160431)
final test f1s (mean, stderr):  (0.1204977549031091, 0.017049426647553059)
final train rates (mean, stderr):  814.131820096 ,  9.2809365895
final train osr2 (mean, stderr):  0.117674600648 ,  0.0132407511201
Confusion matrix for training data:
[[ 0.          0.05335753  0.17495463  0.00145191  0.        ]
 [ 0.          0.01451906  0.09981851  0.00036298  0.        ]
 [ 0.          0.04029038  0.28021779  0.00435572  0.        ]
 [ 0.          0.00907441  0.20108893  0.01052632  0.        ]
 [ 0.          0.00362976  0.09909256  0.00725953  0.        ]]
Confusion matrix for validation data:
[[ 0.          0.02702703  0.18378378  0.00216216  0.        ]
 [ 0.          0.01621622  0.11675676  0.00432432  0.        ]
 [ 0.          0.04216216  0.27783784  0.00540541  0.        ]
 [ 0.          0.01621622  0.19351351  0.00756757  0.        ]
 [ 0.          0.00540541  0.09621622  0.00540541  0.        ]]
Confusion matrix for test data:
[[ 0.          0.03783784  0.18810811  0.00648649  0.        ]
 [ 0.          0.01297297  0.08432432  0.00216216  0.        ]
 [ 0.          0.03783784  0.25837838  0.00540541  0.        ]
 [ 0.          0.01297297  0.22594595  0.00540541  0.        ]
 [ 0.          0.00540541  0.11351351  0.00324324  0.        ]]






INCLUDING:  perception_var
M1
Dropout  0.0 , regul  0.1 :
final val rates (mean, stderr):  261.647462398 ,  5.724167711
final val osr2 (mean, stderr):  0.124979314541 ,  0.0208361229266
final val accs (mean, stderr):  (0.29621621621621619, 0.016869715172245666)
final val f1s (mean, stderr):  (0.15038346968795793, 0.0089019747315082966)
final test rates (mean, stderr):  274.842539063 ,  7.91180027358
final test osr2 (mean, stderr):  0.157816119098 ,  0.0153579527414
final test accs (mean, stderr):  (0.27783783783783783, 0.017144596245571582)
final test f1s (mean, stderr):  (0.13626912083346759, 0.010077350217527333)
final train rates (mean, stderr):  752.991200648 ,  9.06956268586
final train osr2 (mean, stderr):  0.184259840589 ,  0.00526483302746
Confusion matrix for training data:
[[ 0.          0.13212341  0.09038113  0.00725953  0.        ]
 [ 0.          0.02867514  0.08203267  0.00399274  0.        ]
 [ 0.          0.05045372  0.26098004  0.01343013  0.        ]
 [ 0.          0.01742287  0.19128857  0.01197822  0.        ]
 [ 0.          0.01161525  0.07150635  0.02686025  0.        ]]
Confusion matrix for validation data:
[[ 0.          0.10486486  0.09405405  0.01405405  0.        ]
 [ 0.          0.03459459  0.09945946  0.00324324  0.        ]
 [ 0.          0.04756757  0.25081081  0.02702703  0.        ]
 [ 0.          0.0227027   0.18378378  0.01081081  0.        ]
 [ 0.          0.01189189  0.06702703  0.02810811  0.        ]]
Confusion matrix for test data:
[[ 0.          0.13081081  0.09189189  0.00972973  0.        ]
 [ 0.          0.0227027   0.07459459  0.00216216  0.        ]
 [ 0.          0.04216216  0.24648649  0.01297297  0.        ]
 [ 0.          0.02702703  0.20864865  0.00864865  0.        ]
 [ 0.          0.01189189  0.08324324  0.02702703  0.        ]]

INCLUDING:  perception_var
M2
Dropout  0.2 , regul  0.01 :
final val rates (mean, stderr):  264.854549062 ,  4.84494492561
final val osr2 (mean, stderr):  0.114115549455 ,  0.0200857363441
final val accs (mean, stderr):  (0.30486486486486486, 0.010341149443085933)
final val f1s (mean, stderr):  (0.15177556997514213, 0.0051490136550337079)
final test rates (mean, stderr):  278.571903971 ,  8.97344125837
final test osr2 (mean, stderr):  0.146845358251 ,  0.0141184053307
final test accs (mean, stderr):  (0.27783783783783783, 0.014564533605737813)
final test f1s (mean, stderr):  (0.13422320006922855, 0.0099363495339816286)
final train rates (mean, stderr):  753.841272106 ,  10.7191664235
final train osr2 (mean, stderr):  0.183393697625 ,  0.00660836922887
Confusion matrix for training data:
[[ 0.          0.13248639  0.09401089  0.00326679  0.        ]
 [ 0.          0.03266788  0.07949183  0.00254083  0.        ]
 [ 0.          0.04791289  0.27005445  0.00689655  0.        ]
 [ 0.          0.02032668  0.19201452  0.00834846  0.        ]
 [ 0.          0.01161525  0.08203267  0.01633394  0.        ]]
Confusion matrix for validation data:
[[ 0.          0.10486486  0.09837838  0.00972973  0.        ]
 [ 0.          0.03567568  0.09837838  0.00324324  0.        ]
 [ 0.          0.04864865  0.26054054  0.01621622  0.        ]
 [ 0.          0.02594595  0.1827027   0.00864865  0.        ]
 [ 0.          0.01297297  0.06918919  0.02486486  0.        ]]
Confusion matrix for test data:
[[ 0.          0.12648649  0.1027027   0.00324324  0.        ]
 [ 0.          0.02810811  0.07027027  0.00108108  0.        ]
 [ 0.          0.04972973  0.24432432  0.00756757  0.        ]
 [ 0.          0.02810811  0.21081081  0.00540541  0.        ]
 [ 0.          0.01189189  0.09297297  0.0172973   0.        ]]







INCLUDING:  contextual_var
M1
Dropout  0.1 , regul  0.1 :
final val rates (mean, stderr):  280.497508706 ,  11.3715214571
final val osr2 (mean, stderr):  0.0651485981604 ,  0.00751383667325
final val accs (mean, stderr):  (0.27459459459459457, 0.01763189884356766)
final val f1s (mean, stderr):  (0.12056477736524747, 0.0055763621653849623)
final test rates (mean, stderr):  313.720576259 ,  9.04690981906
final test osr2 (mean, stderr):  0.0393442307844 ,  0.00436629733967
final test accs (mean, stderr):  (0.22702702702702701, 0.026035880170361722)
final test f1s (mean, stderr):  (0.095541574492356715, 0.0052985744003387764)
final train rates (mean, stderr):  855.336378467 ,  10.3476850874
final train osr2 (mean, stderr):  0.0734006884022 ,  0.00532792160779
Confusion matrix for training data:
[[ 0.          0.07949183  0.15027223  0.          0.        ]
 [ 0.          0.03194192  0.08275862  0.          0.        ]
 [ 0.          0.07949183  0.24537205  0.          0.        ]
 [ 0.          0.02686025  0.1938294   0.          0.        ]
 [ 0.          0.01633394  0.09364791  0.          0.        ]]
Confusion matrix for validation data:
[[ 0.          0.07027027  0.1427027   0.          0.        ]
 [ 0.          0.03675676  0.10054054  0.          0.        ]
 [ 0.          0.08756757  0.23783784  0.          0.        ]
 [ 0.          0.02162162  0.19567568  0.          0.        ]
 [ 0.          0.02810811  0.07891892  0.          0.        ]]
Confusion matrix for test data:
[[ 0.          0.06162162  0.17081081  0.          0.        ]
 [ 0.          0.0172973   0.08216216  0.          0.        ]
 [ 0.          0.09189189  0.20972973  0.          0.        ]
 [ 0.          0.03135135  0.21297297  0.          0.        ]
 [ 0.          0.01837838  0.10378378  0.          0.        ]]

INCLUDING:  contextual_var
M2
Dropout  0.0 , regul  0.1 :
final val rates (mean, stderr):  280.371686549 ,  10.6864112927
final val osr2 (mean, stderr):  0.0652599605491 ,  0.00451669121988
final val accs (mean, stderr):  (0.30594594594594593, 0.017314180949239973)
final val f1s (mean, stderr):  (0.11011882674110134, 0.0067982326455383428)
final test rates (mean, stderr):  312.237768576 ,  8.2273836573
final test osr2 (mean, stderr):  0.0435573367879 ,  0.00727284336344
final test accs (mean, stderr):  (0.26594594594594589, 0.01570360977982049)
final test f1s (mean, stderr):  (0.095277898522229745, 0.0040006626388095611)
final train rates (mean, stderr):  857.170396712 ,  11.1103011651
final train osr2 (mean, stderr):  0.0714720827351 ,  0.00442429375393
Confusion matrix for training data:
[[ 0.          0.03375681  0.19600726  0.          0.        ]
 [ 0.          0.01306715  0.10163339  0.          0.        ]
 [ 0.          0.03303085  0.29183303  0.          0.        ]
 [ 0.          0.00980036  0.21088929  0.          0.        ]
 [ 0.          0.00580762  0.10417423  0.          0.        ]]
Confusion matrix for validation data:
[[ 0.          0.02918919  0.18378378  0.          0.        ]
 [ 0.          0.01513514  0.12216216  0.          0.        ]
 [ 0.          0.03459459  0.29081081  0.          0.        ]
 [ 0.          0.00756757  0.20972973  0.          0.        ]
 [ 0.          0.01297297  0.09405405  0.          0.        ]]
Confusion matrix for test data:
[[ 0.          0.0172973   0.21513514  0.          0.        ]
 [ 0.          0.00864865  0.09081081  0.          0.        ]
 [ 0.          0.04432432  0.2572973   0.          0.        ]
 [ 0.          0.01513514  0.22918919  0.          0.        ]
 [ 0.          0.00432432  0.11783784  0.          0.        ]]






INCLUDING: sociodemographic_var
M1
Dropout  0.2 , regul  0.1 :
final val rates (mean, stderr):  303.4993121 ,  10.6241556847
final val osr2 (mean, stderr):  -0.012269847729 ,  0.00250996200037
final val accs (mean, stderr):  (0.32324324324324327, 0.0069222964729003704)
final val f1s (mean, stderr):  (0.1002370700555146, 0.0016735409982985986)
final test rates (mean, stderr):  331.754357515 ,  11.5787419709
final test osr2 (mean, stderr):  -0.0150755002502 ,  0.00776318375718
final test accs (mean, stderr):  (0.29621621621621619, 0.012014113930212367)
final test f1s (mean, stderr):  (0.094644298337904859, 0.0034405031392842084)
final train rates (mean, stderr):  913.475104056 ,  10.3434774347
final train osr2 (mean, stderr):  0.0104498485213 ,  0.000841951024617
Confusion matrix for training data:
[[ 0.          0.00290381  0.22686025  0.          0.        ]
 [ 0.          0.00072595  0.11397459  0.          0.        ]
 [ 0.          0.00145191  0.32341198  0.          0.        ]
 [ 0.          0.00036298  0.22032668  0.          0.        ]
 [ 0.          0.00036298  0.10961887  0.          0.        ]]
Confusion matrix for validation data:
[[ 0.          0.          0.21297297  0.          0.        ]
 [ 0.          0.00108108  0.13621622  0.          0.        ]
 [ 0.          0.00324324  0.32216216  0.          0.        ]
 [ 0.          0.00216216  0.21513514  0.          0.        ]
 [ 0.          0.00108108  0.10594595  0.          0.        ]]
Confusion matrix for test data:
[[ 0.          0.00108108  0.23135135  0.          0.        ]
 [ 0.          0.00108108  0.09837838  0.          0.        ]
 [ 0.          0.00648649  0.29513514  0.          0.        ]
 [ 0.          0.00216216  0.24216216  0.          0.        ]
 [ 0.          0.00324324  0.11891892  0.          0.        ]]

INCLUDING:  sociodemographic_var
M2
Dropout  0.1 , regul  0.1 :
final val rates (mean, stderr):  300.854742076 ,  10.0251691242
final val osr2 (mean, stderr):  -0.00368959578849 ,  0.0018168005346
final val accs (mean, stderr):  (0.32540540540540541, 0.0055124535282084089)
final val f1s (mean, stderr):  (0.098184689451296839, 0.0012537234322696131)
final test rates (mean, stderr):  328.326188802 ,  10.7458918787
final test osr2 (mean, stderr):  -0.00482834634998 ,  0.00381627813369
final test accs (mean, stderr):  (0.30162162162162159, 0.011517446352790556)
final test f1s (mean, stderr):  (0.092594372632810312, 0.0027313774983301624)
final train rates (mean, stderr):  918.287001737 ,  10.4391249405
final train osr2 (mean, stderr):  0.00523850046277 ,  0.00114077626865
Confusion matrix for training data:
[[ 0.          0.          0.22976407  0.          0.        ]
 [ 0.          0.          0.11470054  0.          0.        ]
 [ 0.          0.          0.32486388  0.          0.        ]
 [ 0.          0.          0.22068966  0.          0.        ]
 [ 0.          0.          0.10998185  0.          0.        ]]
Confusion matrix for validation data:
[[ 0.          0.          0.21297297  0.          0.        ]
 [ 0.          0.          0.1372973   0.          0.        ]
 [ 0.          0.          0.32540541  0.          0.        ]
 [ 0.          0.          0.2172973   0.          0.        ]
 [ 0.          0.          0.10702703  0.          0.        ]]
Confusion matrix for test data:
[[ 0.          0.          0.23243243  0.          0.        ]
 [ 0.          0.          0.09945946  0.          0.        ]
 [ 0.          0.          0.30162162  0.          0.        ]
 [ 0.          0.          0.24432432  0.          0.        ]
 [ 0.          0.          0.12216216  0.          0.        ]]
"""



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
        if SHOWPLOTS:
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
        else:
            model.fit(train_x, train_y, epochs=num_epochs, batch_size=BATCH_SIZE, verbose=0)
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

overall = []

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
        print("final val rates (mean, stderr): ", np.mean(final_val_rates[:, -1]), ", ", stats.sem(final_val_rates[:, -1]))
        print("final val osr2 (mean, stderr): ", np.mean(final_val_osr2[:, -1]), ", ",  stats.sem(final_val_osr2[:, -1]))
        print("final val accs (mean, stderr): ", get_avg_acc(all_val_y, predict_val))
        print("final val f1s (mean, stderr): ", get_avg_f1(all_val_y, predict_val))
        
        print("final test rates (mean, stderr): ", np.mean(final_test_rates[:, -1]), ", ",  stats.sem(final_test_rates[:, -1]))
        print("final test osr2 (mean, stderr): ", np.mean(final_test_osr2[:, -1]), ", ",  stats.sem(final_test_osr2[:, -1]))
        print("final test accs (mean, stderr): ", get_avg_acc(all_test_y, predict_test))
        print("final test f1s (mean, stderr): ", get_avg_f1(all_test_y, predict_test))
        
        print("final train rates (mean, stderr): ", np.mean(final_train_rates[:, -1]), ", ",  stats.sem(final_train_rates[:, -1]))
        print("final train osr2 (mean, stderr): ", np.mean(final_train_osr2[:, -1]), ", ",  stats.sem(final_train_osr2[:, -1]))

        overall.append((dr, regul, np.mean(final_val_osr2[:, -1])))

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
            plt.title("average sse vs epoch number, no dropout and regularization")
            plt.legend()
            plt.show()

            plt.plot(np.mean(final_train_osr2, axis=0), 'go-', label="train")
            plt.plot(np.mean(final_val_osr2, axis=0), 'ro-', label="val")
            plt.plot(np.mean(final_test_osr2, axis=0), 'bo-', label="test")
            plt.title("average osr2 vs epoch number, no dropout and regularization")
            plt.legend()
            plt.show()

print(overall)
"""
model.fit(final_train, trainclass_y, epochs=50, batch_size=5)
predict_train = model.predict_classes(final_train, verbose=0)
predict_val = model.predict_classes(final_val, verbose=0)
if len(test_y) > 0:
    predict_test = model.predict_classes(final_test, verbose=0)
"""
