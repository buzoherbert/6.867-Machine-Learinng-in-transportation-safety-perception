#Python 3 file

from random import shuffle
import csv
import numpy as np
from sklearn import preprocessing
import math
import keras

trip_var = ["origin", "destination", "companions", "trip_purpose"]
perception_var = ["mode_security", "importance_safety", "most_safe", "least_safe"]
instant_var = ["haversine", "urban_typology", "total_passenger_count", "total_female_count",
                        "empty_seats", "hour_sin", "hour_cos", "week_day"]
contextual_var = ["bus_or_ped", "base_study_zone", "busdestination", "total_seats"]
sociodemographic_var = ["age", "gender", "education"]

class Data:
    def __init__(self, val_ratio, test_ratio, include, filename, normalize = True):
        self.test_ratio = test_ratio #float from 0.0 to 1.0
        self.val_ratio = val_ratio #float from 0.0 to 1.0
        self.normalize = normalize
        if include == "trip_var": #include is one of the variables defined above (trip_var, perception_var, ...)
            self.include = trip_var
        elif include == "perception_var":
            self.include = perception_var
        elif include == "instant_var":
            self.include = instant_var
        elif include == "contextual_var":
            self.include = contextual_var
        elif include == "sociodemographic_var":
            self.include = sociodemographic_var
        elif include == "ALL":
            self.include = None
        else:
            print("include parameter ill-defined")
            raise ValueError("include parameter ill-defined")
            
        self.titles, self.data = self.get_raw_data(filename)
        self.process_data(self.titles, self.data)

    # load data from csv files
    def get_raw_data(self, filename):
        titles = []
        data = []
        with open(filename) as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quotechar='\\')
            row_no = 1
            for row in reader:
                if row_no == 1:
                    row_no += 1
                    titles = [trim_quote(s) for s in row[1:]] #ignore first column
                    continue
                data_row = {}
                for i in range(len(row)-1): #ignore first column
                    data_row[titles[i]] = trim_quote(row[i+1])
                data.append(data_row)
        
        return titles, data

    def process_data(self, titles, data):
        #split train val test
        n = len(data)
        test_no = int(n*self.test_ratio)
        val_no = int(n*self.val_ratio)
        train = data[:n-test_no-val_no]
        val = data[n-test_no-val_no:n-test_no]
        test = data[n-test_no:]

        ptitles = []
        train_plist = []
        val_plist = []
        test_plist = []

        latlontitles = []
        train_latlon = []
        val_latlon = []
        test_latlon = []

        title_y = None
        train_y = None
        val_y = None
        test_y = None

        for title in titles:
            if self.include != None and title not in self.include + ["point_security", "latitude", "longitude"]:
                continue
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
            elif title == "total_passenger_count" or title == "total_female_count" or title == "empty_seats":
                ptitles.append(np.array([title]))
                vec_train = np.array([int(s) for s in train_raw]).reshape((-1, 1))
                vec_val = np.array([int(s) for s in val_raw]).reshape((-1, 1))
                vec_test = np.array([int(s) for s in test_raw]).reshape((-1, 1))
                if self.normalize:
                    vec_train = vec_train/np.max(vec_train)
                    vec_val = vec_val/np.max(vec_val)
                    vec_test = vec_test/np.max(vec_test) if len(vec_test) > 0 else vec_test
                    
                train_plist.append(vec_train)
                val_plist.append(vec_val)
                test_plist.append(vec_test)
            elif title == "hour_sin" or title == "hour_cos" or title == "haversine":
                ptitles.append(np.array([title]))
                vec_train = np.array([float(s) for s in train_raw]).reshape((-1, 1))
                vec_val = np.array([float(s) for s in val_raw]).reshape((-1, 1))
                vec_test = np.array([float(s) for s in test_raw]).reshape((-1, 1))
                if self.normalize and title == "haversine":
                    vec_train = vec_train/np.max(vec_train)
                    vec_val = vec_val/np.max(vec_val)
                    vec_test = vec_test/np.max(vec_test) if len(vec_test) > 0 else vec_test
                    
                train_plist.append(vec_train)
                val_plist.append(vec_val)
                test_plist.append(vec_test)
            elif title == "latitude" or title == "longitude":
                latlontitles.append(np.array([title]))
                vec_train = np.array([float(s) for s in train_raw]).reshape((-1, 1))
                vec_val = np.array([float(s) for s in val_raw]).reshape((-1, 1))
                vec_test = np.array([float(s) for s in test_raw]).reshape((-1, 1))
                train_latlon.append(vec_train)
                val_latlon.append(vec_val)
                test_latlon.append(vec_test)
            else:
                classes, vec_train, vec_val, vec_test = get_vector(train_raw, val_raw, test_raw)
                prefixed = [title + '.' + s for s in classes.tolist()]
                ptitles.append(np.array(prefixed))
                train_plist.append(vec_train)
                val_plist.append(vec_val)
                test_plist.append(vec_test)

        self.final_titles = np.concatenate(ptitles + latlontitles)
        self.final_train = np.concatenate(train_plist + train_latlon, axis=1)
        self.final_val = np.concatenate(val_plist + val_latlon, axis=1)
        self.final_test = np.concatenate(test_plist + test_latlon, axis=1)

        self.title_y = title_y
        self.train_y = train_y
        self.val_y = val_y
        self.test_y = test_y

    def get_title(self):
        return self.final_titles, self.title_y

    def get_train_data(self):
        return self.final_train, self.train_y

    def get_val_data(self):
        return self.final_val, self.val_y

    def get_test_data(self):
        return self.final_test, self.test_y

    def write_to_file(self, filename, datatype):
        if datatype not in ["train", "val", "test"]:
            print("datatype must be either 'train', 'val', or 'test'")
            return
        with open(filename, 'w') as csvfile:
            writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(np.concatenate((self.final_titles, self.title_y)))
            if datatype=="train":
                for i in range(self.train_y.size):
                    writer.writerow(np.concatenate((self.final_train[i], self.train_y[i])))
            elif datatype=="val":
                for i in range(self.val_y.size):
                    writer.writerow(np.concatenate((self.final_val[i], self.val_y[i])))
            elif datatype=="test":
                for i in range(self.test_y.size):
                    writer.writerow(np.concatenate((self.final_test[i], self.test_y[i])))
                    
                



def trim_quote(s):
    if s[0] == '"':
        s = s[1:-1]
    return s


# input are the training/val/test vectors for a single feature
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
