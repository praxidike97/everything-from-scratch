
import numpy as np
from tqdm import tqdm

from NeuralNetwork import NeuralNetwork
from activation_functions import LogisticFunction

def generate_training_examples():
    X = [[0, 1]]*100
    y = [[1, 0]]*100
    return X, y

episodes = range(0, 100)
X, y = generate_training_examples()
net = NeuralNetwork([2, 5, 2], LogisticFunction.logistic_function, 0.001)

for e in tqdm(episodes):
    net.train(X, y)

    error_total = net.calc_total_error(X, y)
    print("Total error: " + str(error_total))

#print(net.calc_feed_forward(np.array([0.05, 0.1])))
#print(net.calc_total_error(np.array([0.05, 0.1]), np.array([0.01, 0.99])))