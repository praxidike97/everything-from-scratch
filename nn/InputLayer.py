
import numpy as np

class InputLayer():

    def __init__(self, number_neurons):
        self.number_neurons = number_neurons

    def calc_feed_forward(self, input):
        self.input = input
        self.output = input
        return np.asarray(self.output)

