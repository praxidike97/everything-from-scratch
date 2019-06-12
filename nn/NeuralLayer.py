
import numpy as np

class NeuralLayer():

    def __init__(self, number_neurons, number_previous_neurons, activation_function, initial_weights='zeros'):
        self.activation_function = activation_function
        #self.delta = 0
        self.stored_delta = []
        self.stored_output = []

        if initial_weights == 'zeros':
            self.weights = np.zeros((number_neurons, number_previous_neurons))
            self.bias = np.zeros((number_neurons, ))
        elif initial_weights == 'random':
            self.weights = np.random.rand(number_neurons, number_previous_neurons)
            self.bias = np.random.rand(number_neurons,)

    def calc_feed_forward(self, input):
        self.input = input
        self.summed_input = np.dot(self.weights, self.input) + self.bias
        self.output = self.activation_function(self.summed_input)
        self.stored_output.append(self.output)
        return np.asarray(self.output)

    def reset(self):
        """
        Remove all stored values (output, delta, etc. )
        :return: None
        """
        self.stored_output = []
        self.stored_delta = []

