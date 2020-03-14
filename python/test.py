import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from keras.layers import Dense
from keras.models import Model, Sequential
from keras.optimizers import Adam, SGD

from activation_functions import LogisticFunction, LinearFunction


def load_data():
    df = pd.read_csv("boston.csv")

    x_raw, y = df.values[:, :-1], df.values[:, -1]
    x = np.asarray([((2*(col - np.min(col)))/(np.max(col) - np.min(col)))-1 for col in x_raw.T]).T

    return x, y


class NeuralNetwork():

    def __init__(self, list_layers):
        self.w = []
        self.a = []
        self.delta = []
        self.activation_functions = []
        self.outputs = []

        for i in range(1, len(list_layers)):
            #self.w.append(np.random.rand(list_layers[i][0], list_layers[i-1][0])*2-1)
            self.w.append(np.random.rand(list_layers[i][0], list_layers[i - 1][0]+1) * 2 - 1)
            self.activation_functions.append(list_layers[i][1])

        for weight in self.w:
            print(weight.shape)

    def forward(self, x):
        self.a = list()
        self.outputs = list()

        self.outputs.append(np.array([x]))
        #x = np.append(x, 1)
        #self.outputs.append(x)

        for i, weights in enumerate(self.w):
            x = np.dot(weights[:, :-1], x)
            x += weights[:, -1]

            #x = np.dot(weights, x)
            #x = np.append(x, 1)
            self.a.append(x)
            x = self.activation_functions[i].function(x)
            self.outputs.append(np.array([x]))

        self.outputs = self.outputs[:-1]
        #self.outputs = self.outputs[1:]

        return x

    def predict(self, xs):
        y_pred = list()

        for x in xs:
            y_pred.append(self.forward(x)[0])

        return np.asarray(y_pred)

    def backpropagate(self, x, y):
        gradients = list()

        # 1. Do a forward pass
        y_pred = self.forward(x=x)

        # 2. Calculate error for output layer
        error = y_pred - y

        # 3. Calculate deltas for all previous layers
        self.delta = list()
        self.delta.append(np.array([error]))
        #self.delta.append(error)

        for i in list(reversed(range(1, len(self.w)))):
            #self.delta.append(self.activation_functions[i-1].function_derivative(self.a[i-1])*np.dot(self.delta[-1].T, self.w[i]))
            self.delta.append(self.activation_functions[i-1].function_derivative(self.a[i-1])*np.dot(self.delta[-1], self.w[i][:, :-1]))

        # 4. Calculate the gradients for all layers
        self.delta = list(reversed(self.delta))
        for i in reversed(range(0, len(self.w))):
            # Gradient for weigth
            gradient = np.dot(self.delta[i].T, self.outputs[i])
            #gradients.append(np.dot(self.delta[i].T, self.outputs[i]))

            # Gradients for bias
            gradient = np.concatenate((gradient, self.delta[i].T), axis=1)
            #gradients.append(np.dot(self.outputs[i], self.delta[i]))

            gradients.append(gradient)

        return list(reversed(gradients)), error

    def train(self, xs, ys, lr=0.001, epochs=1000, batch_size=16):
        for e in range(epochs):

            indices = np.arange(0, len(xs))
            np.random.shuffle(indices)

            xs = xs[indices]
            ys = ys[indices]

            chunked_xs = np.array_split(xs, int(len(xs)/batch_size))
            chunked_ys = np.array_split(ys, int(len(xs)/batch_size))

            for chunk_x, chunk_y in zip(chunked_xs, chunked_ys):
                total_gradient = list()
                for weights in self.w:
                    total_gradient.append(np.zeros((weights.shape[0], weights.shape[1])))

                total_error = 0

                for x, y in zip(chunk_x, chunk_y):
                    gradients, error = self.backpropagate(x, y)
                    #print(gradients)

                    total_error += error
                    for i, g in enumerate(gradients):
                        total_gradient[i] += g

                new_weights = list()
                for gradient, weights in zip(total_gradient, self.w):
                    weights -= lr * (1/len(chunk_x)) * gradient
                    new_weights.append(weights)
                self.w = new_weights

            print("Error in epoch %i: %f" % (e, total_error))


def get_model():
    model = Sequential()
    model.add(Dense(32, activation="sigmoid", input_shape=(13, )))
    model.add(Dense(1, activation="linear"))

    sgd = SGD(lr=0.001)
    model.compile(sgd, loss="mse", metrics=["mse"])
    return model


if __name__ == "__main__":
    x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 0])
    nn = NeuralNetwork([(13, LogisticFunction), (32, LogisticFunction), (1, LinearFunction)])

    x, y = load_data()

    print(nn.predict(x))
    print("RMSE is %f" % (np.sqrt((y - nn.predict(x)) ** 2).mean()))
    print(nn.train(x, y, batch_size=128))
    print("RMSE is %f" % (np.sqrt((y - nn.predict(x))**2).mean()))

    model = get_model()
    model.fit(x, y, epochs=1000, batch_size=128, verbose=0)
    print("RMSE is %f" % (np.sqrt((y - model.predict(x))**2).mean()))

    ax = plt.gca()
    #line1 = ax.plot(y, label="True data")
    line2 = ax.plot(nn.predict(x), label="My model")
    line3 = ax.plot(model.predict(x), label="Keras model")
    ax.legend()
    plt.show()