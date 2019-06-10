
import numpy as np

class Neuron:

    def __init__(self, activation_function):
        self.activation_function = activation_function

    def calc_feed_forward(self, input):
        return self.activation_function(input)