#Python 3 file

from random import shuffle
import csv
import numpy as np
#from nn import NeuralNet
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation #Dropout, Flatten
#from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras import initializers
from sklearn import preprocessing
import math
import matplotlib.pyplot as plt

#np.random.seed(0)
TEST_RATIO = 0.0 #float from 0.0 to 1.0
VAL_RATIO = 0.2 #float from 0.0 to 1.0
NORMALIZE = True #normalize data in "total_passenger_count", "total_female_count", "empty_seats", "haversine"
BATCH_SIZE = 5

print('======Training======')
# load data from csv files
def trim_quote(s):
    if s[0] == '"':
        s = s[1:-1]
    return s

titles = []
data = []

with open('safety_data_clean.csv') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar='\\')
    row_no = 1
    for row in reader:
        if row_no == 1:
            row_no += 1
            titles = [trim_quote(s) for s in row[1:]] #ignore first column
            continue
        if row[11] == '"I"': #ignore the weird row with I as importance
            continue
        data_row = {}
        for i in range(len(row)-1): #ignore first column
            data_row[titles[i]] = trim_quote(row[i+1])
        data.append(data_row)

#if one of "total_passenger_count", "total_female_count", or "empty_seats" is negative, set them all to 0
for point in data:
    x = min(int(point["total_passenger_count"]), int(point["total_female_count"]), int(point["empty_seats"]))
    if x < 0:
        point["total_passenger_count"] = 0
        point["total_female_count"] = 0
        point["empty_seats"] = 0


#split train val test
shuffle(data)
n = len(data)
test_no = int(n*TEST_RATIO)
val_no = int(n*VAL_RATIO)
train = data[:n-test_no-val_no]
val = data[n-test_no-val_no:n-test_no]
test = data[n-test_no:]


# process all categories
def get_vector(train, val, test):
    le = preprocessing.LabelEncoder()
    train_classes = le.fit_transform(train)
    num_classes = len(le.classes_)
    val_classes = [le.transform([s])[0] if s in le.classes_ else num_classes for s in val] #create new "other" category for unseen
    test_classes = [le.transform([s])[0] if s in le.classes_ else num_classes for s in test]

    vector = keras.utils.to_categorical(np.concatenate([train_classes, np.array(val_classes + test_classes)]))
    vector_trim = vector[:,:num_classes] #trim away "other" category

    return le.classes_, vector_trim[:len(train)], vector_trim[len(train):len(train)+len(val)], vector_trim[len(train)+len(val):]

ptitles = []
train_plist = []
val_plist = []
test_plist = []

title_y = None
train_y = None
val_y = None
test_y = None

for title in titles:
    train_raw = [s[title] for s in train]
    val_raw = [s[title] for s in val]
    test_raw = [s[title] for s in test]
    """
    if title == "age":
        cat_to_no = {"0-17": 9, "18-24": 21, "25-44": 35, "45-64": 55, "65+": 65}
        ptitles.append(title)
        train_plist.append(np.array([cat_to_no[s] for s in train_raw]).reshape((-1, 1)))
        val_plist.append(np.array([cat_to_no[s] for s in val_raw]).reshape((-1, 1)))
        test_plist.append(np.array([cat_to_no[s] for s in test_raw]).reshape((-1, 1)))
    """
    if title == "mode_security" or title == "importance_safety":
        ptitles.append(np.array([title]))
        train_plist.append(np.array([0.2*int(s) for s in train_raw]).reshape((-1, 1))) #0.2 factor to normalize
        val_plist.append(np.array([0.2*int(s) for s in val_raw]).reshape((-1, 1)))
        test_plist.append(np.array([0.2*int(s) for s in test_raw]).reshape((-1, 1)))
    elif title == "point_security":
        title_y = np.array([title])
        train_y = np.array([int(s)-1 for s in train_raw]).reshape((-1, 1)) #convert from 1-5 to 0-4
        val_y = np.array([int(s)-1 for s in val_raw]).reshape((-1, 1))
        test_y = np.array([int(s)-1 for s in test_raw]).reshape((-1, 1))
    elif title == "total_passenger_count" or title == "total_female_count" or title == "empty_seats" or title == "haversine":
        ptitles.append(np.array([title]))
        vec_train = np.array([int(s) for s in train_raw]).reshape((-1, 1))
        vec_val = np.array([int(s) for s in val_raw]).reshape((-1, 1))
        vec_test = np.array([int(s) for s in test_raw]).reshape((-1, 1))
        if NORMALIZE:
            vec_train = vec_train/np.max(vec_train)
            vec_val = vec_val/np.max(vec_val)
            vec_test = vec_test/np.max(vec_test) if len(vec_test) > 0 else vec_test
            
        train_plist.append(vec_train)
        val_plist.append(vec_val)
        test_plist.append(vec_test)
    elif title == "hour":
        train_sin = [math.sin(2*math.pi*int(s)/24.0) for s in train_raw]
        train_cos = [math.cos(2*math.pi*int(s)/24.0) for s in train_raw]
        val_sin = [math.sin(2*math.pi*int(s)/24.0) for s in val_raw]
        val_cos = [math.cos(2*math.pi*int(s)/24.0) for s in val_raw]
        test_sin = [math.sin(2*math.pi*int(s)/24.0) for s in test_raw]
        test_cos = [math.cos(2*math.pi*int(s)/24.0) for s in test_raw]
        
        ptitles.append(np.array(["hour_sin", "hour_cos"]))
        train_plist.append(np.array([train_sin, train_cos]).transpose())
        val_plist.append(np.array([val_sin, val_cos]).transpose())
        test_plist.append(np.array([test_sin, test_cos]).transpose())
    else:
        classes, vec_train, vec_val, vec_test = get_vector(train_raw, val_raw, test_raw)
        prefixed = [title + '.' + s for s in classes.tolist()]
        ptitles.append(np.array(prefixed))
        train_plist.append(vec_train)
        val_plist.append(vec_val)
        test_plist.append(vec_test)

final_titles = np.concatenate(ptitles)
final_train = np.concatenate(train_plist, axis=1)
final_val = np.concatenate(val_plist, axis=1)
final_test = np.concatenate(test_plist, axis=1)

trainclass_y = keras.utils.to_categorical(train_y)
valclass_y = []
testclass_y = []
"""
with open('safety_data_vectorized_train.csv', 'w') as csvfile:
    writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(np.concatenate((final_titles, title_y)))
    for i in range(train_y.size):
        writer.writerow(np.concatenate((final_train[i], train_y[i])))

with open('safety_data_vectorized_val.csv', 'w') as csvfile:
    writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(np.concatenate((final_titles, title_y)))
    for i in range(val_y.size):
        writer.writerow(np.concatenate((final_val[i], val_y[i])))
"""
#Y_train_class = keras.utils.to_categorical(Y_modified, num_classes=2)

M1 = 12
M2 = 10
input_dim = len(final_titles)

model = Sequential()
model.add(Dense(M1, activation='relu', input_dim=input_dim))
if M2 != None:
    model.add(Dense(M2, activation='relu'))
model.add(Dense(5))
model.add(Activation('softmax'))

#sgd = SGD(lr=0.01)
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

train_rates = []
val_rates = []
test_rates = []

for i in range(50):
    model.fit(final_train, trainclass_y, epochs=1, batch_size=BATCH_SIZE)
    predict_train = model.predict_classes(final_train, verbose=0)
    train_rates.append(get_error_rate(train_y, predict_train))
    predict_val = model.predict_classes(final_val, verbose=0)
    val_rates.append(get_error_rate(val_y, predict_val))
    if len(test_y) > 0:
        predict_test = model.predict_classes(final_test, verbose=0)
        test_rates.append(get_error_rate(test_y, predict_test))

train_confusion = np.zeros((5, 5))
val_confusion = np.zeros((5, 5))
test_confusion = np.zeros((5, 5))
for i in range(len(train_y)):
    train_confusion[train_y[i][0]][predict_train[i]] += 1
for i in range(len(val_y)):
    val_confusion[val_y[i][0]][predict_val[i]] += 1
for i in range(len(test_y)):
    test_confusion[test_y[i][0]][predict_test[i]] += 1
print("Confusion matrix for training data:")
print(train_confusion)

print("Confusion matrix for validation data:")
print(val_confusion)

print("Confusion matrix for test data:")
print(test_confusion)


plt.plot(train_rates, 'go-', label="train")
plt.plot(val_rates, 'ro-', label="val")
plt.plot(test_rates, 'bo-', label="test")
plt.title("error rate vs epoch number")
plt.legend()
plt.show()

