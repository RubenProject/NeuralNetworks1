import numpy as np
import csv
import math
import random

#run with python task5_MNIST.py


def read_csv():
    with open("../data/test_in.csv", 'rb') as f:
        reader = csv.reader(f)
        test_in = list(reader)
        test_in = [map(float, i) for i in test_in]

    with open("../data/test_out.csv", 'rb') as f:
        reader = csv.reader(f)
        test_out = list(reader)
        test_out = [map(float, i) for i in test_out]
        test_out = [i for j in test_out for i in j]

    with open("../data/train_in.csv", 'rb') as f:
        reader = csv.reader(f)
        train_in = list(reader)
        train_in = [map(float, i) for i in train_in]


    with open("../data/train_out.csv", 'rb') as f:
        reader = csv.reader(f)
        train_out = list(reader)
        train_out = [map(float, i) for i in train_out]
        train_out = [i for j in train_out for i in j]
    return test_in, test_out, train_in, train_out


#linear rectifier
def g(x):
    return max(0,x)


def MNIST_net(X, W):
    #256 + 1 inputs
    # 30 + 1 hidden
    # 10 outputs
    #257 * 30 + 31 * 10 weights
    #
    p = 0 #weight index
    X = [-1] + X
    H = [-1] + [0] * 30
    Y = [0] * 10
    for i in range(1, 31):
        for j in range(257):
            H[i] += X[j] * W[p]
            p += 1
        H[i] = g(H[i])

    for i in range(10):
        for j in range (31):
            Y[i] += H[j] * W[p]
            p += 1
        Y[i] = g(Y[i])


    return Y


def mse(D_in, D_out, W):
    MSE = 0
    for e, f in zip(D_in, D_out):
        y_pred = MNIST_net(e, W) 
        for i in range(0, 10):
            if i == f:
                MSE += pow(1 - y_pred[i], 2)
            else:
                MSE += pow(y_pred[i], 2)

    #calculate mean
    MSE /= len(D_out) * 10
    return MSE


def grdmse(D_in, D_out, W):
    e = 10e-3

    gw = [0] * len(W)
    w0 = mse(D_in, D_out, W)
    for i in range(len(W)):
        W[i] = W[i] + e
        gw[i] = (mse(D_in, D_out, W) - w0) / e
        W[i] = W[i] - e
    return gw


def res (L):
    highest = -100
    index = -1
    for i in range(10):
        if L[i] > highest:
            highest = L[i]
            index = i
    return index


#calculate accuracy on testset
def test(D_in, D_out, W):
    c = 0.0
    for x, y in zip(D_in, D_out):
        if res(MNIST_net(x, W)) == y:
            c += 1
    return c / len(D_out)


def randIndx(length, size):
    r = random.randint(0, length - 1 - size)
    upper = r + size
    lower = r
    return lower, upper


def train(D_in, D_out, W, eta, epochs):
    sample_size = 50 
    for i in range(0, epochs):        
        #update w
        print "epoch {}".format(i)
        [lower , upper] = randIndx(len(D_out), sample_size)
        sub_D_in = D_in[lower:upper]
        sub_D_out = D_out[lower:upper]
        gw = grdmse(sub_D_in, sub_D_out, W)
        for j in range (wlen):
            W[j] = W[j] - eta * gw[j]
    return W
            
    

wlen = 257 * 30 + 31 * 10 #total amount of weights
print wlen
#read test/train
[test_in, test_out, train_in, train_out] = read_csv()
W = [random.normalvariate(0.0, 4) for _ in range(wlen)]

f = open("out.txt", "w")

for i in range(5):
    W = train(train_in, train_out, W, 0.1, 10)
    acc = test(test_in, test_out, W)
    f.write(str(i) + ", acc = " + str(acc) + "\n")

f.close()
