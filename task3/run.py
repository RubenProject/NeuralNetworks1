import numpy as np
import csv
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn import datasets


def get_group(label, L_in, L_out):
    e = np.array(L_out, dtype=float)
    ei = (e == label).nonzero()
    L = list() 
    for i in ei[0]:
        L.append(L_in[int(i)])
    return L


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


def get_energy(L):
    e = 0
    for i in L:
        e += i
    return e

def get_avg_ratio(L):
    avg_ratio = 0.0
    H = list()
    for i in L:
        t = get_xy_ratio(i)
        avg_ratio += t
        H.append(t)
    L_ratio = avg_ratio / len(L)
    fig = plt.figure()
    plt.xlabel('x:y ratio')
    plt.ylabel('Frequency')
    plt.hist(H, bins = 20)
    return L_ratio


def get_avg_energy(L):
    avg_e = 0.0
    H = list()
    for i in L:
        t = get_energy(i)
        avg_e += t
        H.append(t)
    L_e = avg_e / len(L)
    fig = plt.figure()
    plt.xlabel('Energy')
    plt.ylabel('Frequency')
    plt.hist(H, bins = 20)
    return L_e


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


#inputs - labels - average energy a / b - target a / b
def eval_by_feature(train_in, train_out, e_a, e_b, t_a, t_b, get_feature):
    count = 0.0
    for i, j in zip(train_in, train_out):
        e = get_feature(i)
        if abs(e - e_a) <= abs(e - e_b):
            if j == t_a:
                count += 1
        elif j == t_b:
            count += 1
    return count / len(train_out)


def get_label(label, L):
    ret = list()
    for i in L:
        if i == label:
            ret.append(i)
    return ret



def main ():
    [test_in, test_out, train_in, train_out] = read_csv()

    Train1 = get_group(1.0, train_in, train_out)
    Train8 = get_group(8.0, train_in, train_out)
    Train1_8 = Train1 + Train8

    Test1 = get_group(1.0, test_in, test_out)
    Test8 = get_group(8.0, test_in, test_out)
    Test1_8 = Test1 + Test8

    TrainTar1 = get_label(1.0, train_out)
    TrainTar8 = get_label(8.0, train_out)
    TrainTar1_8 = TrainTar1 + TrainTar8 

    TestTar1 = get_label(1.0, test_out)
    TestTar8 = get_label(8.0, test_out)
    TestTar1_8 = TestTar1 + TestTar8 

    #analysis
    print ("P(C) = " + str(float(len(Train1)) / (len(Train1_8))))

#train
#=========================================================
    feat = list()
    for i in Train1_8:
        feat.append(get_xy_ratio(i))

    gnb = GaussianNB()
    x = np.array(feat) 
    x = x.reshape(-1, 1)
    y = np.array(TrainTar1_8)
    y_pred = gnb.fit(x, y).predict(x)

    print (y_pred == TrainTar1_8).sum()
    t = (y_pred == 1.0).sum()
    print "P(X)= " + str(float(t) /len(TrainTar1_8))

#=========================================================
    feat = list()
    for i in Test1_8:
        feat.append(get_xy_ratio(i))

    x = np.array(feat)
    x = x.reshape(-1, 1)
    y_pred = gnb.predict(x)

    #print (y_pred != TestTar1_8).sum()

#=========================================================
    feat = list()
    for i in Train1:
        feat.append(get_xy_ratio(i))

    x = np.array(feat)
    x = x.reshape(-1, 1)
    y_pred = gnb.predict(x)

    count = 0
    for i, j in zip(y_pred, TrainTar1):
        if (i == j):
            count += 1
    print "P(X|C)= " + str(float(count) / len(Train1))





    #print ("P(X|C_1) =" + str(eval_by_feature(T1, tar1, r1, r8, 1.0, 8.0, get_xy_ratio)))
    #print ("P(X|C_8) =" + str(eval_by_feature(T8, tar8, r1, r8, 1.0, 8.0, get_xy_ratio)))
    #plt.show()

main()


