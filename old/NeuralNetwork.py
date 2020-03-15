
import numpy as np
from tqdm import tqdm

from NeuralLayer import NeuralLayer
from InputLayer import InputLayer
from error_functions import MSE
from activation_functions import LogisticFunction


class NeuralNetwork:
    """
    Main Sources:
    http://neuralnetworksanddeeplearning.com/chap2.html
    https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
    """

    def __init__(self, list_layers, activation_function, learning_rate, batch_size=32):
        self.list_layers = list_layers
        self.learning_rate = learning_rate
        self.layers = []
        self.batch_size = batch_size

        self.layers.append(InputLayer(list_layers[0]))
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

    def reset(self):
        """
        Reset stored values in all layers
        :return: None
        """
        for layer in self.layers:
            layer.reset()

    def split_in_chunks(self, list, chunksize):
        return [list[i * chunksize:(i + 1) * chunksize] for i in range((len(list) + chunksize - 1) // chunksize)]

    def train(self, Xs, ys):

        batched_X = self.split_in_chunks(Xs, chunksize=self.batch_size)
        batched_y = self.split_in_chunks(ys, chunksize=self.batch_size)

        for X_batch, y_batch in tqdm(zip(batched_X, batched_y), total=len(batched_X)):

            # 1. Forward propagation and error calculation
            #error_total = self.calc_total_error(X, y)
            #print("Total error: " + str(error_total))

            # Compute the error for batch_size examples

            for X, y in zip(X_batch, y_batch):
                # 1. Feedforward
                self.calc_feed_forward(X)

                # 2. Compute the error for the last layer
                cost_derivative_final_layer = MSE.mse_derivative(self.layers[-1].output, y)
                activation_derivative_final_layer = LogisticFunction.function_derivative(self.layers[-1].summed_input)
                self.layers[-1].delta = np.multiply(cost_derivative_final_layer, activation_derivative_final_layer)
                self.layers[-1].stored_delta.append(np.multiply(cost_derivative_final_layer, activation_derivative_final_layer))

                # 3. Compute the error for all the other layers
                for i in range(2, len(self.layers)):
                    weighted_error = np.dot(self.layers[-i+1].weights.T, self.layers[-i+1].delta)
                    activation_derivative = LogisticFunction.function_derivative(self.layers[-i].summed_input)
                    self.layers[-i].stored_delta.append(np.multiply(weighted_error, activation_derivative))
                    self.layers[-i].delta = np.multiply(weighted_error, activation_derivative)

            # 4. Change the weights and biases in all the layers according to the calculated delta
            for i in range(1, len(self.layers)):

                sum_delta = np.zeros((self.layers[i].stored_delta[0].shape[0]))
                sum_delta_dot_output = np.zeros((self.layers[i].stored_delta[0].shape[0], self.layers[i-1].stored_output[0].shape[0]))

                for output, delta in zip(self.layers[i-1].stored_output, self.layers[i].stored_delta):
                    output_last_layer = np.reshape(output, (1, output.shape[0]))
                    delta_current_layer = np.reshape(delta, (delta.shape[0], 1))

                    sum_delta += np.squeeze(delta_current_layer)
                    sum_delta_dot_output += np.dot(delta_current_layer, output_last_layer)

                #output_last_layer = np.reshape(self.layers[i-1].output, (1, self.layers[i-1].output.shape[0]))
                #delta_current_layer = np.reshape(self.layers[i].delta, (self.layers[i].delta.shape[0], 1))

                #self.layers[i].weights -= self.learning_rate*(np.dot(delta_current_layer, output_last_layer))
                #self.layers[i].bias -= self.learning_rate *self.layers[i].delta

                self.layers[i].weights -= (1/float(self.batch_size))*self.learning_rate*sum_delta_dot_output
                self.layers[i].bias -= (1/float(self.batch_size))*self.learning_rate*sum_delta

            self.reset()
