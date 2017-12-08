import csv
import numpy as np

FILENAME = "example.csv"

confusions = []

with open(FILENAME) as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar='\\')
    confusion = np.zeros((5, 5))
    i = 0
    for row in reader:
        confusion[i] = row
        i += 1
        if i % 5 == 0:
            i = 0
            confusions.append(confusion)
            confusion = np.zeros((5, 5))

def get_total_mtx(confusions):
    total = np.sum(np.stack(confusions), axis=0)
    return total/total.sum()

overall = get_total_mtx(confusions)
print(FILENAME[:-4])
print(overall)
