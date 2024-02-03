import numpy as np

# static classes for activation functions linear, relu, sigmoid, tanh and their derivatives

class linear(object):
    @staticmethod
    # linear function
    def g(x):
        return x * np.ones(len(x))
    
    @staticmethod
    # linear differentiation function
    def g_diff(x):
        return np.ones(len(x))
    

class relu(object):

    @staticmethod
    # rectified linear function
    def g(x):
        return np.array(list(map(lambda i: max(i,0), x)))
    
    @staticmethod
    # rectified linear differentiation function
    def g_diff(x):
        return np.array(list(map(lambda i: 1 if i > 0 else 0, x))) 
    
    
class sigmoid(object):
    @staticmethod
    # sigmoid function
    def g(x):
        return np.array(list(map(lambda i: 1 / (1 + np.exp(-1 * i)), x)))   
    
    @staticmethod
    # sigmoid differentiation function
    def g_diff(x):
        return np.multiply(sigmoid.g(x) , np.ones(len(x)) - sigmoid.g(x))
    

class tanh(object):
    @staticmethod
    # tanh function
    def g(x):
        return np.array(list(map(lambda i: np.tanh(i), x))) 
    
    @staticmethod
    # tanh differentiation function
    def g_diff(x):
        return np.multiply((1 / np.cosh(x)), (1 / np.cosh(x)))
    