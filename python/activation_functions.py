
import numpy as np
import scipy.special

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
        #return 1
        return np.ones(len(output))


class SoftMax(ActivationFunction):

    @staticmethod
    def function(output):
        #return np.array([(np.exp(x)/np.sum(np.exp(output))) for x in output])
        return scipy.special.softmax(output)

    @staticmethod
    # For derivative of SoftMax see:
    # https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/ and
    # https://stackoverflow.com/questions/40575841/numpy-calculate-the-derivative-of-the-softmax-function
    def function_derivative(output):
        derivatives = np.zeros(len(output))
        for i in range(len(output)):
            for j in range(len(output)):
                if i == j:
                    softmax = np.exp(output[i]/np.sum(np.exp(output)))
                    derivatives[i] -= softmax *(1 - softmax)
                else:
                    derivatives[i] += (np.exp(output[i])*np.exp(output[j]))/np.sum(np.exp(output))**2
        return derivatives