
import numpy as np

class InputLayer():

    def __init__(self, number_neurons):
        self.number_neurons = number_neurons
        self.stored_output = []

    def calc_feed_forward(self, input):
        self.input = input
        self.output = input
        self.stored_output.append(self.output)
        return np.asarray(self.output)

    def reset(self):
        self.stored_output = []