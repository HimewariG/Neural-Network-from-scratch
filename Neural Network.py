import numpy as np
import matplotlib
import random
layers = 4
b1 = np.random.rand(2)
b2 = np.random.rand(3)
b3 = np.random.rand(2)
w1 = np.random.rand(2,2)
w2 = np.random.rand(3,2)
w3 = np.random.rand(2,3)
nb = int(input("Please enter the nuber of data point "))
train_rate = float(input("Please enter the training rate "))
# for i in range(nb): #still working on input
training_input = []
label = []
for i in range(nb):
    training_data= np.zeros(2)
    training_data[0] = float(input("please enter training data accordingly,the current input is for input, first dimension"))
    training_data[1] = float(input("please enter training data accordingly,the current input is for input,second dimension"))
    training_input.append(training_data)
    training_lab = np.zeros(2)
    training_lab[0] = float(input("please enter training data accordingly,the current input is for input, first dimension"))
    training_lab[1] = float(input("please enter training data accordingly,the current input is for input,second dimension"))
    label.append(training_lab)
def sigma(x):
    return 1/(1+np.exp(-x))
def d_sigma(x):
    return sigma(x)*(1 - sigma(x))

def forward(x,weight,bias):
    a = [None] * 4
    z = [None] * 4
    a[0] = x
    for i in range(1, 4):
        z[i] = np.matmul(weight[i], a[i - 1]) + bias[i]
        a[i] = sigma(z[i])
    return a[3]
def backward(x,y,weight,bias):

    # weight = [None,w1, w2, w3]
    # bias = [None,b1,b2,b3]
    delta = [None]*4
    a = [None]*4
    D = [None]*4
    z = [None]*4
    a[0] = x
    for i in range (1,4):
        z[i] = np.matmul(weight[i] , a[i-1])+bias[i]
        a[i] = sigma(z[i])
        D[i] = np.diag(d_sigma(z[i]))
    delta[3] = np.matmul(D[3],(a[3] - y))
    for i in range(2, 0, -1):
        delta[i] = np.matmul(np.matmul(D[i],weight[i+1].T),delta[i+1])
    return delta, a


def training():
    weight = [None, w1, w2, w3]
    bias = [None, b1, b2, b3]
    for i in range(nb):
        k = random.randint(0, nb-1)
        delta, a = backward(training_input[k],label[k],weight,bias)
        for j in range(3,0,-1):
            weight[j] = weight[j] - train_rate*np.outer(delta[j],a[j-1])
            bias[j] = bias[j] - train_rate * delta[j]
    return weight,bias

def Neural_network():
    weight,bias = training()
    data = float(input("please enter your data point that you want to determine"))
    output = forward(data,weight,bias)
    print("output is",output)

Neural_network()