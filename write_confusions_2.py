
import csv
import numpy as np

matrices_acc = []
matrices_f1 = []

with open('logistic_confusions.txt') as file:
    i = 0
    rows = []
    for line in file:
        line = line.strip()
        if len(line) < 1:
            continue
        if line[-1] == ":":
            i += 1
            continue
        line = line.replace("[", "").replace("]", "")
        numbers = line.split()
        rows.append(numbers)
        if len(rows) == 5:
            if i == 1:
                matrices_acc.append(rows)
            elif i == 2:
                matrices_f1.append(rows)
            rows = []

def transpose(mat):
    new_mat = [[row[i] for row in mat] for i in range(len(mat))]
    return new_mat

with open('confusions_logistic_acc.csv', 'w') as csvfile:
    writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    for mat in matrices_acc:
        new_mat = transpose(mat)
        for row in new_mat:
            writer.writerow(row)

with open('confusions_logistic_f1.csv', 'w') as csvfile:
    writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    for mat in matrices_f1:
        new_mat = transpose(mat)
        for row in new_mat:
            writer.writerow(row)
