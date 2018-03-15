import numpy as np
import math
import random
import sys

#sigmoid
def g(x):
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
    
           
def train(init_w, learn_rate):
    w = init_w
    eta = learn_rate
    epochs = 50000
        
    for i in range(0, epochs):        
        #update w
        gw = grdmse(w)
        for j in range (0, 9):
            w[j] = w[j] - eta * gw[j]
            
        #print mse(w) & error rate for every 500 upgrades of w
        if i%5000 == 0:
            error_nr = 0
            for k in range(0,1000):
                result = xor_net(train_X1[k], train_X2[k],w)
                if result > 0.5:
                    pred_class = 1
                else:
                    pred_class = 0
                if pred_class != train_class[k]:
                    error_nr += 1
            print(f'weights update: {i}th, mse: {mse(w)}, misclassification: {error_nr}') 
    
    #return the prediction accuracy of the final weights
    correct = 0
    for k in range(0,1000):
                result = xor_net(train_X1[k], train_X2[k],w)
                if result > 0.5:
                    pred_class = 1
                else:
                    pred_class = 0
                if pred_class == train_class[k]:
                    correct += 1
    return(correct/1000)
    print(w)



#generate train set, size = 1000
train_X1 = np.random.randint(2, size=1000)
train_X2 = np.random.randint(2, size=1000)
def xor(x1, x2):
    if x1 != x2:
        return 1
    else:
        return 0

train_class = [0]*1000
for i in range(0,1000):
    train_class[i] = xor(train_X1[i], train_X2[i])



#test with differnt activation function
#f1=sigmoid
#f2=hyperbolic tangent
#f3=linear rectifier

#test with different random initalization
#r1=random draws from unif(-1,1)
#r2=random draws from unif(-4,4)
#r3=random draws from norm(0,1)
#r4=random draws from norm(0,4)

#test with different learning rate(eta)
#e1=0.1
#e2=0.01

#3*4*2 = 24combos


f = open("output.txt", "w")
#c1.sigmoid, r1, e1
t = 0
for i in range(0, 5):
    t = train([random.uniform(-1,1) for _ in range(9)], 0.1)
t /= 5
f.write(f"r1, e1, {t}\n")


#c2.sigmoid, r1, e2
t = 0
for i in range(0, 5):
    t = train([random.uniform(-1,1) for _ in range(9)], 0.01)
t /= 5
f.write(f"r1, e2, {t}\n")


#c3.sigmoid, r2, e1
t = 0
for i in range(0, 5):
    t = train([random.uniform(-4,4) for _ in range(9)], 0.1)
t /= 5
f.write(f"r2, e1, {t}\n")


#c4.sigmoid, r2, e2
t = 0
for i in range(0, 5):
    t = train([random.uniform(-4,4) for _ in range(9)], 0.01)
t /= 5
f.write(f"r2, e2, {t}\n")


#c5.sigmoid, r3, e1
t = 0
for i in range(0, 5):
    t = train(np.random.normal(0,1,9), 0.1)
t /= 5
f.write(f"r3, e1, {t}\n")


#c6.sigmoid, r3, e2
t = 0
for i in range(0, 5):
    t = train(np.random.normal(0,1,9), 0.01)
t /= 5
f.write(f"r3, e2, {t}\n")


#c7.sigmoid, r4, e1
t = 0
for i in range(0, 5):
    t = train(np.random.normal(0,4,9), 0.1)
t /= 5
f.write(f"r4, e1, {t}\n")


#c8.sigmoid, r4, e2
t = 0
for i in range(0, 5):
    t = train(np.random.normal(0,4,9), 0.01)
t /= 5
f.write(f"r4, e2, {t}\n")


#hyperbolic tangent
g = np.tanh


#c9.hyperbolic tangent, r1, e1
t = 0
for i in range(0, 5):
    t = train([random.uniform(-1,1) for _ in range(9)], 0.1)
t /= 5
f.write(f"r1, e1, {t}\n")


#c10.hyperbolic tangent, r1, e2
t = 0
for i in range(0, 5):
    t = train([random.uniform(-1,1) for _ in range(9)], 0.01)
t /= 5
f.write(f"r1, e2, {t}\n")


#c11.hyperbolic tangent, r2, e1
t = 0
for i in range(0, 5):
    t = train([random.uniform(-4,4) for _ in range(9)], 0.1)
t /= 5
f.write(f"r2, e1, {t}\n")


#c12.hyperbolic tangent, r2, e2
t = 0
for i in range(0, 5):
    t = train([random.uniform(-4,4) for _ in range(9)], 0.01)
t /= 5
f.write(f"r2, e2, {t}\n")



#c13.hyperbolic tangent, r3, e1
t = 0
for i in range(0, 5):
    t = train(np.random.normal(0,1,9), 0.1)
t /= 5
f.write(f"r3, e1, {t}\n")


#c14.hyperbolic tangent, r3, e2
t = 0
for i in range(0, 5):
    t = train(np.random.normal(0,1,9), 0.01)
t /= 5
f.write(f"r3, e2, {t}\n")


#c15.hyperbolic tangent, r4, e1
t = 0
for i in range(0, 5):
    t = train(np.random.normal(0,4,9), 0.1)
t /= 5
f.write(f"r4, e1, {t}\n")


#c16.hyperbolic tangent, r4, e2
t = 0
for i in range(0, 5):
    t = train(np.random.normal(0,4,9), 0.01)
t /= 5
f.write(f"r4, e2, {t}\n")


#linear rectifier
def g(x):
    return max(0,x)


#c17.linear rectifier, r1, e1
t = 0
for i in range(0, 5):
    t = train([random.uniform(-1,1) for _ in range(9)], 0.1)
t /= 5
f.write(f"r1, e1, {t}\n")


#c18.linear rectifier, r1, e2
t = 0
for i in range(0, 5):
    t = train([random.uniform(-1,1) for _ in range(9)], 0.01)
t /= 5
f.write(f"r1, e2, {t}\n")


#c19.linear rectifier, r2, e1
t = 0
for i in range(0, 5):
    t = train([random.uniform(-4,4) for _ in range(9)], 0.1)
t /= 5
f.write(f"r2, e1, {t}\n")


#c20.linear rectifier, r2, e2
t = 0
for i in range(0, 5):
    t = train([random.uniform(-4,4) for _ in range(9)], 0.01)
t /= 5
f.write(f"r2, e2, {t}\n")


#c21.linear rectifier, r3, e1
t = 0
for i in range(0, 5):
    t = train(np.random.normal(0,1,9), 0.1)
t /= 5
f.write(f"r3, e1, {t}\n")


#c22.linear rectifier, r3, e2
t = 0
for i in range(0, 5):
    t = train(np.random.normal(0,1,9), 0.01)
t /= 5
f.write(f"r3, e2, {t}\n")


#c23.linear rectifier, r4, e1
t = 0
for i in range(0, 5):
    t = train(np.random.normal(0,4,9), 0.1)
t /= 5
f.write(f"r4, e1, {t}\n")


#c24.linear rectifier, r4, e2
t = 0
for i in range(0, 5):
    t = train(np.random.normal(0,4,9), 0.01)
t /= 5
f.write(f"r4, e2, {t}\n")



f.close()
