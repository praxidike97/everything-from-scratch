
import numpy as np
from tqdm import tqdm
import random

from NeuralNetwork import NeuralNetwork
from activation_functions import LogisticFunction
from data.data_utils import get_mnist_dataset

def generate_training_examples():
    X = [[0, 1]]*25
    y = [[0, 1]]*25

    X += [[1, 0]]*25
    y += [[1, 0]] * 25

    X += [[0, 0]]*25
    y += [[0,0]] * 25

    X += [[1, 1]]*25
    y += [[1,1]] * 25

    random.seed(4)
    random.shuffle(X)
    random.seed(4)
    random.shuffle(y)

    return np.asarray(X), np.asarray(y)

episodes = range(0, 50)
#X, y = generate_training_examples()
X_train, X_test, y_train, y_test = get_mnist_dataset()
print(X_train[2])
print(y_train[2])
net = NeuralNetwork([784, 1024, 10], LogisticFunction.logistic_function, 0.001, batch_size=512)

for e in tqdm(episodes):
    net.train(X_train, y_train)

    error_total = net.calc_total_error(X_train[1], y_train[1])
    print("Total error: " + str(error_total))

print(net.calc_feed_forward([0, 0]))
print(net.calc_feed_forward([0, 1]))
print(net.calc_feed_forward([1, 0]))
print(net.calc_feed_forward([1, 1]))

#print(net.calc_feed_forward(np.array([0.05, 0.1])))
#print(net.calc_total_error(np.array([0.05, 0.1]), np.array([0.01, 0.99])))