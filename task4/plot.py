import matplotlib.pyplot as plt
import csv

#this program is solely used to plot data from the network
#run with python plot.py


with open("./data.txt", 'rb') as f:
    reader = csv.reader(f)
    data = list(reader)
    data = [map(float, i) for i in data]

print data


data0 = list()
data1 = list()
data2 = list()
for i in data:
    data0.append(i[0])
    data1.append(i[1])
    data2.append(i[2])

fig = plt.figure()
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.plot(data0, data1, '-r', data0, data2, '-b')
plt.show()
