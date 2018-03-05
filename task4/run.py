import numpy as np
import math
import csv



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


def g(x):
    return 1 / (1 + math.exp(-x))

def g_prime(x):
    return g(x) * (1- g(x))



def train_net(t_in, t_out, epochs):
    eta = 0.01
    n_input = len(t_in[0]) #nr of inputs
    n_hidden = 10 #nr of hidden nodes
    n_output = 10 #nr of outputs
    hidden_w = np.random.rand(n_input, n_hidden) #10 by 10
    hidden_in = np.random.rand(n_hidden) #10 hidden nodes
    hidden_out = np.random.rand(n_hidden) #10 hidden nodes
    out_w = np.random.rand(n_hidden, n_output) #10 by 10
    out_in = np.random.rand(n_output) #10 classes
    out_out = np.random.rand(n_output) #10 classes

    error = np.random.rand(n_output)
    delta = np.random.rand(n_output)
    deltahidden = np.random.rand(n_hidden, n_output)
#TODO ADD BIASS NODES
    for i in range(0, epochs):
        print "epoch: " + str(i)
        for j, row in enumerate(t_in):
            target = int(t_out[j])
            for k in range(0, n_hidden):
                hidden_in[k] = 0
                for l in range(0, n_input):
                    hidden_in[k] += row[l] * hidden_w[l][k]
            for k in range(0, n_hidden):
                hidden_out[k] = g(hidden_in[k])
            for k in range(0, n_output):
                out_in[k] = 0
                for l in range(0, n_hidden):
                    out_in[k] = hidden_out[l] * out_w[l][k]
            for k in range(0, n_output):
                out_out[k] = g(out_in[k])
            #calculate the error for each output node
            for k in range(0, n_output):
                error[k] = 1
            error[target] = 1.0 - out_out[target]
            #calculate the delta for each output node
            for k in range(0, n_output):
                delta[k] = error[k] * g_prime(out_in[k])
                for l in range(0, n_hidden):
                    deltahidden[k][l] = g_prime(hidden_in[l]) * out_w[l][k] * delta[k]
            #update weights
            for k in range(0, n_output):
                for l in range(0, n_hidden):
                    out_w[l][k] = out_w[l][k] + eta * hidden_out[l] * delta[k]
            for k in range(0, n_output):
                for l in range(0, n_hidden):
                    for m in range(0, n_input):
                        hidden_w[m][l] = hidden_w[m][l] + eta * row[m] * deltahidden[k][l]


    return hidden_w.tolist(), out_w.tolist()


def test_net(hidden_w, output_w, t_in, t_out):
    n_input = len(t_in[0]) #nr of inputs
    n_hidden = 10 #nr of hidden nodes
    n_output = 10 #nr of outputs
    hidden_in = np.random.rand(n_hidden) #10 hidden nodes
    hidden_out = np.random.rand(n_hidden) #10 hidden nodes
    out_in = np.random.rand(n_output) #10 classes
    out_out = np.random.rand(n_output) #10 classes

    mse = np.zeros(n_output)
    for i, row in enumerate(t_in):
        target = int(t_out[i])
        for j in range(0, n_hidden):
            hidden_in[j] = 0
            for k in range(0, n_input):
                hidden_in[j] += row[k] * hidden_w[k][j]
        for j in range(0, n_hidden):
            hidden_out[j] = g(hidden_in[j])
        for j in range(0, n_output):
            outout_in[j] = 0
            for k in range(0, n_hidden):
                outout_in[j] += outout_w[k][j]
        for j in range(0, n_output):
            out_out[j] = g(out_in[j])
            error = abs(target - out_out[j])
            mse[j] += error * error

    return mse


def main ():
    [test_in, test_out, train_in, train_out] = read_csv()
    [h_w, o_w] = train_net(train_in, train_out, 10)
    print h_w
    print "-----------------------------------"
    print o_w
    print test_net(h_w, o_w, test_in, test_out)

main()
