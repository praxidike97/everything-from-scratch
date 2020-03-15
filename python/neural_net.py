import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt

from activation_functions import LogisticFunction, LinearFunction, SoftMax
from error_functions import MSE


def load_data(path):
    df = pd.read_csv(path)
    x_raw, y = df.values[:, :-1], df.values[:, -1]
    x = np.asarray([((2*(col - np.min(col)))/(np.max(col) - np.min(col)))-1 for col in x_raw.T]).T
    return x, y


def numeric_to_onehot(array):
    array = array.astype("int")
    onehot = np.zeros((array.size, array.max() + 1))
    onehot[np.arange(array.size), array] = 1
    return onehot


def onehot_to_numeric(array):
    return np.argmax(array, axis=1)


def accuracy(xs01, xs02):
    if len(xs01) != len(xs02):
        return -1.0
    else:
        return sum([1. if x01 == x02 else 0. for x01, x02 in zip(xs01.tolist(), xs02.tolist())])/len(xs01)


class NeuralNetwork():

    def __init__(self, list_layers):
        # List with matrices of weights
        self.w = []

        # List of summed inputs
        self.a = []

        # Lists od errors
        self.delta = []

        # List of activation functions per layer
        self.activation_functions = []

        # List of outputs per layer
        self.outputs = []
        self.output_last_layer = []

        for i in range(1, len(list_layers)):
            # Add one additional dimension for the bias
            self.w.append(np.random.rand(list_layers[i][0], list_layers[i - 1][0]+1) * 2 - 1)
            self.activation_functions.append(list_layers[i][1])

    def forward(self, x):
        self.a = list()
        self.outputs = list()
        self.outputs.append(np.array([x]))

        for i, weights in enumerate(self.w):
            # Don't multiply with the last dimension, because that's the bias
            x = np.dot(weights[:, :-1], x)

            # Add the bias
            x += weights[:, -1]

            self.a.append(x)
            x = self.activation_functions[i].function(x)
            self.outputs.append(np.array([x]))

        # Drop the last output layer
        self.outputs = self.outputs[:-1]
        self.output_last_layer = self.outputs[-1]

        return x

    def predict(self, xs):
        y_pred = list()
        for x in xs:
            y_pred.append(self.forward(x))

        return np.squeeze(np.asarray(y_pred))

    def backpropagate(self, x, y, error_function=MSE):
        gradients = list()

        # 1. Do a forward pass
        y_pred = self.forward(x=x)

        # 2. Calculate error for output layer
        error = error_function.function_derivative(y_pred, y)

        # 3. Calculate deltas for the output and all previous layers
        self.delta = list()
        self.delta.append(np.array([error * self.activation_functions[-1].function_derivative(y_pred)]))

        for i in list(reversed(range(1, len(self.w)))):
            self.delta.append(self.activation_functions[i-1].function_derivative(self.a[i-1])*np.dot(self.delta[-1], self.w[i][:, :-1]))

        # 4. Calculate the gradients for all layers
        self.delta = list(reversed(self.delta))
        for i in reversed(range(0, len(self.w))):
            # Gradient for weight
            gradient = np.dot(self.delta[i].T, self.outputs[i])

            # Gradients for bias
            gradient = np.concatenate((gradient, self.delta[i].T), axis=1)

            gradients.append(gradient)

        return list(reversed(gradients)), error

    def train(self, xs, ys, lr=0.01, epochs=1000, batch_size=16, error_function=MSE, metrics=[]):
        for e in range(epochs):

            # Shuffle the data
            indices = np.arange(0, len(xs))
            np.random.shuffle(indices)

            xs = xs[indices]
            ys = ys[indices]

            # Split the data into chunks
            chunked_xs = np.array_split(xs, int(len(xs)/batch_size))
            chunked_ys = np.array_split(ys, int(len(xs)/batch_size))

            for chunk_x, chunk_y in zip(chunked_xs, chunked_ys):

                # Create empty matrices for the weights
                total_gradient = list()
                for weights in self.w:
                    total_gradient.append(np.zeros((weights.shape[0], weights.shape[1])))

                total_error = 0

                # Iterate over the data chunks
                for x, y in zip(chunk_x, chunk_y):
                    gradients, error = self.backpropagate(x, y, error_function=error_function)

                    # Add all the gradients
                    total_error += np.sum(abs(error))
                    for i, g in enumerate(gradients):
                        total_gradient[i] += g

                new_weights = list()
                for gradient, weights in zip(total_gradient, self.w):
                    # Update the weights
                    weights -= lr * (1/len(chunk_x)) * gradient
                    new_weights.append(weights)
                self.w = new_weights

            print("Error in epoch %i: %f" % (e, np.mean(total_error)))
            if "accuracy" in metrics:
                print("Accuracy: %f" % (accuracy(np.argmax(self.predict(xs), axis=1), onehot_to_numeric(ys))))


if __name__ == "__main__":
    # Create neural net
    nn = NeuralNetwork([(4, LogisticFunction), (8, LogisticFunction), (3, SoftMax)])

    # Load the (normalized) data
    x, y = load_data(path="../data/iris.csv")
    y = numeric_to_onehot(y)

    # Train the neural net and print the error
    nn.train(x, y, batch_size=4, epochs=500)
    print(np.argmax(nn.predict(x), axis=1))
    print(nn.predict(x))
    diff = y - nn.predict(x)
    preds = nn.predict(x)
    print("RMSE is %f" % (np.sqrt((y - nn.predict(x))**2).mean()))

    # Plot the results
    ax = plt.gca()
    ax.plot(y, label="True data")
    ax.plot(nn.predict(x), label="My model")
    ax.legend()
    plt.show()
