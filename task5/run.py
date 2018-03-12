import numpy as np
import math
import random
import sys




#sigmoid
def g (x):
    return 1 / (1 + math.exp(-x))


def xor_net(x1, x2, w):
    h1 = 1 * w[0] + x1 * w[1] + x2 * w[2]
    h2 = 1 * w[3] + x1 * w[4] + x2 * w[5]
    h1 = g(h1)
    h2 = g(h2)
    y = 1 * w[6] + h1 * w[7] + h2 * w[8]
    return g(y)


def mse(weights):
    se_00 = (xor_net(0, 0, weights) - 0)
    se_00 = se_00 * se_00
    se_01 = (xor_net(0, 1, weights) - 1)
    se_01 = se_01 * se_01
    se_10 = (xor_net(1, 0, weights) - 1)
    se_10 = se_10 * se_10
    se_11 = (xor_net(1, 1, weights) - 0)
    se_11 = se_11 * se_11
    mean = se_00 + se_01 + se_10 + se_11
    mean = mean / 4
    return(mean)


def grdmse(w):
    e = 10e-3
    gw = [0] * 9
    w0 = mse(w)
    for i in range(0, 9):
        w[i] = w[i] + e
        gw[i] = (mse(w) - w0) / e
        w[i] = w[i] - e
    return gw
    
           
def main ():
    eta = 0.1
#possibly use different random initalization?
    w = [random.uniform(-1, 1) for _ in xrange(9)]
    if len(sys.argv) > 1:
        epochs = int(sys.argv[1])
    else:
        epochs = 10000
    for i in range(0, epochs):
        gw = grdmse(w)
        for j in range (0, 9):
            w[j] = w[j] - eta * gw[j]
        print mse(w)

main()
