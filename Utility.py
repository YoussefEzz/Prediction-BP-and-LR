import numpy as np

class utility(object):
    @staticmethod
    # linear function
    def linear(x):
        return x

    @staticmethod
    # rectified linear function
    def relu(x):
        return max(0.0, x)
    
    @staticmethod
    # sigmoid function
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
  
    @staticmethod
    # tanh function
    def tanh(x):
        return np.tanh(x)