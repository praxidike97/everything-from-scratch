
import numpy as np


class ActivationFunction:

    @staticmethod
    def function(output):
        pass

    @staticmethod
    def function_derivative(output):
        pass


class LogisticFunction(ActivationFunction):

    @staticmethod
    def function(output):
        return 1/(1 + np.e**(-output))

    @staticmethod
    def function_derivative(output):
        x = LogisticFunction.function(output)
        return x*(1 - x)


class LinearFunction(ActivationFunction):

    @staticmethod
    def function(output):
        return output

    @staticmethod
    def function_derivative(output):
        return 1
