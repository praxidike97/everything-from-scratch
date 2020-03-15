

class ErrorFunction:

    @staticmethod
    def function(output, target):
        pass

    @staticmethod
    def function_derivative(output, target):
        pass


class MSE(ErrorFunction):
    @staticmethod
    def function(output, target):
        return 0.5*(output-target)**2

    @staticmethod
    def function_derivative(output, target):
        return output - target
