import numpy as np
import csv


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
    for i in L:
        avg_ratio += get_xy_ratio(i)
    L_ratio = avg_ratio / len(L)
    return L_ratio


def get_avg_energy(L):
    avg_e = 0.0
    for i in L:
        avg_e += get_xy_ratio(i)
    L_e = avg_e / len(L)
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
    L1 = get_group(1.0, test_in, test_out)
    L8 = get_group(8.0, test_in, test_out)

    r1 = get_avg_ratio(L1)
    r8 = get_avg_ratio(L8)

    e1 = get_avg_energy(L1)
    e8 = get_avg_energy(L8)

    T1 = get_group(1.0, train_in, train_out)
    T8 = get_group(8.0, train_in, train_out)
    T1_8 = T1 + T8

    tar1 = get_label(1.0, train_out)
    tar8 = get_label(8.0, train_out)
    tar1_8 = tar1 + tar8

    print eval_by_feature(T1_8, tar1_8, e1, e8, 1.0, 8.0, get_energy)
    print eval_by_feature(T1_8, tar1_8, r1, r8, 1.0, 8.0, get_xy_ratio)

main()
