import os
from PIL import Image
from math import exp
import random
import numpy as np
from matplotlib import pyplot

alphabet_to_int = {
    'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'J': 9
}

def sigmoid(x):
    return 1/(1+np.exp(-x))

def d_sigmoid(x):
    return x*(1-x)

def get_sample_data2(folder_name):
    X = np.zeros((200, 784))
    Y = np.zeros((200, 10))
    path = os.getcwd() + '/' + folder_name
    for idx, folder_name in enumerate(os.listdir(path)):
        folder_path = path + '/' + folder_name
        for index, image_name in enumerate(os.listdir(folder_path)):
            if index == 1220:
                break
            if index >= 1200:   
                image_path = folder_path + '/' + image_name
                image = Image.open(image_path)
                image = image.convert('1')
                data = [0 for i in range(10)]
                data[alphabet_to_int[folder_name]] = 1
                Y[idx * 10 + index-1200] = data
                pixels = image.load()
                data = []
                for i in range(28):
                    for j in range(28):
                        data.append(0 if pixels[i,j] == 0 else 1)
                X[idx * 10 + index-1200] = data
    return X, Y

def get_sample_data(folder_name):
    X = np.zeros((3000, 784))
    Y = np.zeros((3000, 10))
    path = os.getcwd() + '/' + folder_name
    for idx, folder_name in enumerate(os.listdir(path)):
        folder_path = path + '/' + folder_name
        for index, image_name in enumerate(os.listdir(folder_path)):
            if index == 300:
                break
            image_path = folder_path + '/' + image_name
            image = Image.open(image_path)
            image = image.convert('1')
            data = [0 for i in range(10)]
            data[alphabet_to_int[folder_name]] = 1
            Y[idx * 300 + index] = data
            pixels = image.load()
            data = []
            for i in range(28):
                for j in range(28):
                    data.append(0 if pixels[i,j] == 0 else 1)
            X[idx * 300 + index] = data
    return X, Y

def calculate_loss(o, y):
    return np.mean(np.square(y - o))

def accuracy(o, y):
    s = 0
    n = len(o)
    t = 1 / float(n)
    for i in range(n):
        s += t if np.argmax(o[i]) == np.argmax(y[i]) else 0
    return s

def stochastic_gradient_descent(i, input_data, result, w_hidden, b_hidden, w_output, b_output, do_dropout, h, r, dropout_percent, lr):
    for index, data in enumerate(input_data):
        data2 = np.zeros((1, 784))
        data2[0] = data
        result2 = np.zeros((1, 10))
        result2[0] = result[index]
        # forward propagation
        hidden_result = sigmoid(data2.dot(w_hidden) + b_hidden)
        output_result = hidden_result.dot(w_output) + b_output

        # backward propagation
        d_output = result2 - output_result
        d_hidden = d_output.dot(w_output.T) * d_sigmoid(hidden_result)
        # dropout regularization
        if(do_dropout):
            d_hidden *= np.random.binomial([np.ones((len(data2),h))],1-dropout_percent)[0] * (1.0/(1-dropout_percent))
        w_output += (hidden_result.T.dot(d_output) + w_output * r) * lr
        w_hidden += (data2.T.dot(d_hidden) + w_hidden * r) * lr
        b_output += d_output.sum(axis=0) * lr
        b_hidden += d_hidden.sum(axis=0) * lr
    print calculate_loss(output_result, result)

def gradient_descent(l, i, input_data, result, w_hidden, b_hidden, w_output, b_output, do_dropout, h, r, dropout_percent, lr):
     # forward propagation
    hidden_result = sigmoid(input_data.dot(w_hidden) + b_hidden)
    output_result = hidden_result.dot(w_output) + b_output

    # backward propagation
    d_output = result - output_result
    d_hidden = d_output.dot(w_output.T) * d_sigmoid(hidden_result)
    # dropout regularization
    if(do_dropout):
        d_hidden *= np.random.binomial([np.ones((len(input_data),h))],1-dropout_percent)[0] * (1.0/(1-dropout_percent))
    w_output += (hidden_result.T.dot(d_output) + w_output * r) * lr
    w_hidden += (input_data.T.dot(d_hidden) + w_hidden * r) * lr
    b_output += d_output.sum(axis=0) * lr
    b_hidden += d_hidden.sum(axis=0) * lr
    loss = calculate_loss(output_result, result)
    print loss
    if i % 100 == 0:
        l.append((i, loss))
    return l

def main():

    # initializing
    h = 15 # hidden layer size
    n = 2000 # epoch
    lr = 0.00001 # learning rate
    r = 0.1 # regularization
    dropout_percent = 0.1 # dropout
    do_dropout = True # dropout
    w_hidden = np.random.uniform(-0.01, 0.01, (784, h))
    b_hidden = np.zeros(h)
    w_output = np.random.uniform(-0.01, 0.01, (h, 10))
    b_output = np.zeros(10)
    l = []

    # getting input
    input_data, result = get_sample_data('notMNIST_small')

    # training NN
    for i in range(n):
        # stochastic_gradient_descent(i, input_data, result, w_hidden, b_hidden, w_output, b_output, do_dropout, h, r, dropout_percent, lr)
        gradient_descent(l, i, input_data, result, w_hidden, b_hidden, w_output, b_output, do_dropout, h, r, dropout_percent, lr)
        print "iteration: " + str(i + 1)
    lx = []
    ly = []
    for i in l:
        lx.append(i[0])
        ly.append(i[1])
    print lx, ly
    pyplot.plot(lx, ly)
    pyplot.show()
    
    # testing NN
    X, Y = get_sample_data2('notMNIST_small')
    hidden_result = sigmoid(X.dot(w_hidden) + b_hidden)
    output_result = hidden_result.dot(w_output) + b_output
    print "Accuracy: " + str(accuracy(output_result, Y) * 100) + "%"

if __name__ == '__main__':
    main()
