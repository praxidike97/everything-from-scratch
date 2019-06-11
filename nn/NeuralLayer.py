
import numpy as np

class NeuralLayer():

    def __init__(self, number_neurons, number_previous_neurons, activation_function, initial_weights='zeros'):
        self.activation_function = activation_function

        self.delta = 0
        #for _ in range(0, number_neurons):
        #    self.neurons.append(Neuron(logistic_function))

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
        #output = []
        #for i, neuron in enumerate(self.neurons):
        #    output.append(neuron.calc_feed_forward(input=summed_input[i]))
        return np.asarray(self.output)

