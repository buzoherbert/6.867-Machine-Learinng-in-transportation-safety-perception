import random
import csv
import math

random.seed(0)

titles = []
data = []
with open('safety_data_clean_latlon.csv') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar='\\')
    row_no = 1
    for row in reader:
        if row_no == 1:
            row_no += 1
            titles = row[:23] + ['"hour_sin"', '"hour_cos"'] + row[24:]
            continue
        if row[11] == '"I"': #ignore the weird row with I as importance
            continue
        a = int(row[20]) # total_pass_count
        b = int(row[21]) # total_fem_count
        c = int(row[22]) # empty_seats

        if min(a, b, c) < 0:
            row[20:23] = [0, 0, 0]

        hour = int(row[23][1:-1]) # hour
        hour_sin = '{0:f}'.format(math.sin(2*math.pi*hour/24.0))
        hour_cos = '{0:f}'.format(math.cos(2*math.pi*hour/24.0))

        new_row = row[:23] + [hour_sin, hour_cos] + row[24:]

        data.append(new_row)

for i in range(5):
    random.shuffle(data)
    with open('final_data_' + str(i) + '.csv', 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(titles)
        for d in data:
            writer.writerow(d)
