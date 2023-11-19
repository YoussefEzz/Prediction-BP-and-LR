import pandas as pd
import numpy as np

# Neural Network class
class MyNeuralNetwork:

  def __init__(self, layers):
    self.L = len(layers)    # number of layers
    self.n = layers.copy()  # number of neurons in each layer

    self.theta = [] #an array of arrays for the thresholds (θ)
    for lay in range(self.L):
      self.theta.append(np.zeros(layers[lay]))

    self.xi = []            # node values
    for lay in range(self.L):
      self.xi.append(np.zeros(layers[lay]))

    self.w = []             # edge weights
    self.w.append(np.zeros((1, 1)))
    for lay in range(1, self.L):
      self.w.append(np.zeros((layers[lay], layers[lay - 1])))

    self.h = []            # nan array of arrays for the fields (h)
    for lay in range(self.L):
      self.h.append(np.zeros(layers[lay]))
  
  
  # the activation function that will be used, using function references. 
  def fact(self, operation, x):
    return operation(x)
  
  # X : an array of arrays size (n_samples,n_features), which holds the training samples represented as floating point
  #feature vectors; and a vector y of size (n_samples), which holds the target
  #values for the training samples
  def fit(self, X, y):

    return 0

#read and parse the .csv features file 
df = pd.read_csv('Normalized Data/A1-turbine_normalized.txt', delimiter = '\t')
df.head()

columns = df.shape[1]

# construct an array of arrays size (451, 4)
inputcolumns = df.columns[0 : 4]
features = df[inputcolumns].values

# construct an array of size (451)
outputcolumn = df.columns[4]
targets = df[outputcolumn].values

# layers include input layer + hidden layers + output layer
layers = [4, 9, 5, 1]
nn = MyNeuralNetwork(layers)

# call fit function with features (n_samples,n_features) and targets (n_samples)
nn.fit(features, targets)

print("L = ", nn.L, end="\n")
print("n = ", nn.n, end="\n")

print("xi = ", nn.xi, end="\n")
print("xi[0] = ", nn.xi[0], end="\n")
print("xi[1] = ", nn.xi[0], end="\n")

print("wh = ", nn.w, end="\n")
print("wh[1] = ", nn.w[1], end="\n")
