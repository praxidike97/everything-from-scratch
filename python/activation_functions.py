
import numpy as np


class LogisticFunction:

    @staticmethod
    def logistic_function(output):
        return 1/(1 + np.e**(-output))

    @staticmethod
    def logistic_function_derivative(output):
        x = LogisticFunction.logistic_function(output)
        return x*(1 - x)
