import numpy as np

# static classes for activation functions

class linear(object):
    @staticmethod
    # linear function
    def g(x):
        return x
    
    @staticmethod
    # linear differentiation function
    def g_diff(x):
        return 1
    

class relu(object):

    @staticmethod
    # rectified linear function
    def g(x):
        return max(0.0, x) 
    
    @staticmethod
    # rectified linear differentiation function
    def g_diff(x):
        return 1 if x >= 0 else 0
    
class sigmoid(object):
    @staticmethod
    # sigmoid function
    def g(x):
        return 1 / (1 + np.exp(-x))
    
    @staticmethod
    # sigmoid differentiation function
    def g_diff(x):
        return sigmoid.g(x) * sigmoid.g(1 - x)
    
class tanh(object):
    @staticmethod
    # tanh function
    def g(x):
        return np.tanh(x)
    
    @staticmethod
    # tanh differentiation function
    def g_diff(x):
        return (1 / np.cosh(x)) * (1 / np.cosh(x))