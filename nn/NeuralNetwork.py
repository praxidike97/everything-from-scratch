
import numpy as np

from NeuralLayer import NeuralLayer
from error_functions import MSE
from activation_functions import LogisticFunction


class NeuralNetwork:
    """
    Main Sources:
    http://neuralnetworksanddeeplearning.com/chap2.html
    https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
    """

    def __init__(self, list_layers, activation_function, learning_rate):
        self.list_layers = list_layers
        self.learning_rate = learning_rate
        self.layers = []

        for i in range(1, len(list_layers)):
            self.layers.append(NeuralLayer(list_layers[i], list_layers[i-1], activation_function, initial_weights='random'))

        """
        layer01 = NeuralLayer(2, 2, activation_function)
        layer01.weights = np.array([[0.15, 0.2], [0.2, 0.3]])
        layer01.bias = np.array([0.35, 0.35])

        layer02 = NeuralLayer(2, 2, activation_function)
        layer02.weights = np.array([[0.4, 0.45], [0.5, 0.55]])
        layer02.bias = np.array([0.6, 0.6])

        self.layers.append(layer01)
        self.layers.append(layer02)
        """

    def calc_feed_forward(self, input):
        output = np.copy(input)
        for layer in self.layers:
            output = layer.calc_feed_forward(output)
        return output

    def calc_total_error(self, input, target):
        output = self.calc_feed_forward(input=input)
        return np.sum(MSE.mse(output, target))

    def train(self, Xs, ys):
        for X, y in zip(Xs, ys):

            # 1. Forward propagation and error calculation
            error_total = self.calc_total_error(X, y)
            #print("Total error: " + str(error_total))

            # 2. Compute the error for the last layer
            cost_derivative_final_layer = MSE.mse(self.layers[-1].output, y)
            activation_derivative_final_layer = LogisticFunction.logistic_function_derivative(self.layers[-1].summed_input)
            self.layers[-1].delta = np.multiply(cost_derivative_final_layer, activation_derivative_final_layer)

            # 3. Compute the error for all the other layers
            for i in range(2, len(self.layers)):
                weighted_error = np.dot(self.layers[-i+1].weights.T, self.layers[-i+1].delta)
                activation_derivative = LogisticFunction.logistic_function_derivative(self.layers[-i].summed_input)
                self.layers[-i].delta = np.multiply(weighted_error, activation_derivative)

            # 4. Change the weights and biases in all the layers according to the calculated delta
            for i in range(1, len(self.layers)):
                self.layers[i].weights -= self.learning_rate*(np.dot(self.layers[i-1].output, self.layers[i].delta))
                self.layers[i].weights -= self.learning_rate *self.layers[i].delta
