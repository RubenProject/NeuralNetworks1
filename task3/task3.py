
# coding: utf-8

# In[203]:


import numpy as np
import csv
import matplotlib.pyplot as plt
import os
import pandas as pd

def get_xy_ratio(L):
    A = np.array(L)
    A = A.reshape(16, 16)

    #find y
    y_min = -1
    y_max = -1
    for i, e in enumerate(A):
        if (e > 0.0).sum() > 0:
            if y_min == -1:
                y_min = i
            y_max = i

    #find x
    A = A.transpose()
    x_min = -1
    x_max = -1
    for i, e in enumerate(A):
        if (e > 0.0).sum() > 0:
            if x_min == -1:
                x_min = i
            x_max = i

    dx = x_max - x_min
    dy = y_max - y_min
    ratio = float(dx) / float(dy)
    return ratio



# In[204]:


#import as dataframe
train_in = pd.read_csv('../data/train_in.csv', header = None)
train_out = pd.read_csv('../data/train_out.csv', header = None)
test_in = pd.read_csv('../data/test_in.csv', header = None)
test_out = pd.read_csv('../data/test_out.csv', header = None)

#get index of the vectors of integer1, train
idx_1 = train_out.index[train_out.iloc[:,0] == 1].tolist()
#get index of the vectors of integer8, train
idx_8 = train_out.index[train_out.iloc[:,0] == 8].tolist()

#input vectors of 1&8, train
trainin_1 = train_in.iloc[idx_1,:]
trainin_8 = train_in.iloc[idx_8,:]
trainin_18 = trainin_1.append(trainin_8)
#...and the corresponding labels, train
trainout_1 = train_out.iloc[idx_1,:]
trainout_8 = train_out.iloc[idx_8,:]
trainout_18 = trainout_1.append(trainout_8)

#get index of the vectors of integer1, test
idx_1_test = test_out.index[test_out.iloc[:,0] == 1].tolist()
#get index of the vectors of integer8, test
idx_8_test = test_out.index[test_out.iloc[:,0] == 8].tolist()

#input vectors of 1&8, test
testin_1 = test_in.iloc[idx_1_test,:]
testin_8 = test_in.iloc[idx_8_test,:]
testin_18 = testin_1.append(testin_8)
#...and the corresponding labels, test
testout_1 = test_out.iloc[idx_1_test,:]
testout_8 = test_out.iloc[idx_1_test,:]
testout_18 = testout_1.append(testout_8)


# In[205]:


#prior probabilities P(C1), P(C8)
P_C1 = float(len(idx_1))/(float(len(idx_1)) + float(len(idx_8)))
P_C8 = float(len(idx_8))/(float(len(idx_1)) + float(len(idx_8)))
print(P_C1, P_C8)

#conditional probabilities P(X|C1), P(X|C2)
#1)get xy ratios of the samples, each for C1 and C8
nr1 = len(idx_1)
xy_ratios_1 = [0] * nr1
for i in range(0,nr1):
    xy_ratios_1[i] = get_xy_ratio(trainin_1.iloc[i,:])
xy_ratios_1

nr8 = len(idx_8)
xy_ratios_8 = [0] * nr8
for i in range(0,nr8):
    xy_ratios_8[i] = get_xy_ratio(trainin_8.iloc[i,:])
xy_ratios_8

#2)convert X as a categorical variable with 15 categories
bins = np.linspace(0,1,15)
def num_to_cat(num):
    if num < bins[1]:
        return("x1")
    elif num < bins[2]:
        return("x2")
    elif num < bins[3]:
        return("x3")
    elif num < bins[4]:
        return("x4")
    elif num < bins[5]:
        return("x5")
    elif num < bins[6]:
        return("x6")
    elif num < bins[7]:
        return("x7")
    elif num < bins[8]:
        return("x8")
    elif num < bins[9]:
        return("x9")
    elif num < bins[10]:
        return("x10")
    elif num < bins[11]:
        return("x11")
    elif num < bins[12]:
        return("x12")
    elif num < bins[13]:
        return("x13")
    elif num < bins[14]:
        return("x14")
    else:
        return("x15")

X_c1 = [num_to_cat(item) for item in xy_ratios_1]
X_c8 = [num_to_cat(item) for item in xy_ratios_8]

#3)get P(X|C1), P(X|C2) for X=x1,x2,....,x15
def condc1(x):
    return(X_c1.count(x)/nr1)
def condc8(x):
    return(X_c8.count(x)/nr8)

#bayes classifier
def bayesclass(l):
    n = get_xy_ratio(l)
    x = num_to_cat(n)
    
    #P(C1|X) = P(X|C1) * P(C1)
    postc1 = condc1(x)*P_C1
    #P(C8|X) = P(X|C8) * P(C8)
    postc2 = condc8(x)*P_C8
    
    if postc1>postc2:
        return(1)
    else:
        return(8)


# In[206]:


#classify trainset with bayes classifier, see the prediction accuracy
correct = 0
for i in range(0, trainin_18.shape[0]):
    pred = bayesclass(trainin_18.iloc[i,:])
    if pred == trainout_18.iloc[i,0]:
        correct += 1
print('prediction accuracy, train set: ', correct/trainin_18.shape[0])

#classify testset with bayes classifier, see the prediction accuracy
correct = 0
for i in range(0, testin_18.shape[0]):
    pred = bayesclass(testin_18.iloc[i,:])
    if pred == testout_18.iloc[i,0]:
        correct += 1
print('prediction accuracy, test set: ', correct/testin_18.shape[0])


# In[207]:


#histogram of P(X|C1) and P(X|C8)
from matplotlib import pyplot

weights1 = np.ones_like(xy_ratios_1)/len(xy_ratios_1)
weights8 = np.ones_like(xy_ratios_8)/len(xy_ratios_8)

pyplot.hist(xy_ratios_1, bins, alpha = 0.5, color = "red", histtype = "bar", ec = 'black', label = "P(X|C1)", weights = weights1)
pyplot.hist(xy_ratios_8, bins, alpha = 0.5, color = "blue", histtype = "bar", ec = 'black', label = "P(X|C8)", weights = weights8)
pyplot.legend(loc='upper right')
pyplot.xlabel("value of X")
pyplot.ylabel("P(X|C)")
pyplot.show()

