
class MSE:
    @staticmethod
    def mse(output, target):
        return 0.5*(output-target)**2

    @staticmethod
    def mse_derivative(output, target):
        return output - target
