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
    num_training_patterns = X.shape[0]
    for i in range(0 , num_training_patterns):
      self.feedforward(X[i])
    return 0
  
  #Feed−forward propagation of pattern xµ to obtain the output o(xµ)
  def feedforward(self, x):
    y = x
    return 0

#read and parse the .csv features file 
df = pd.read_csv('Normalized Data/A1-turbine_normalized.txt', delimiter = '\t')
df.head()

columns = df.shape[1]

# construct an array of arrays size (451, 4) for all features input values
inputcolumns = df.columns[0 : 4]
features = df[inputcolumns].values

#select the first 85% as training features an array of arrays size (383, 4)
num_training_features = int(85 * features.shape[0] / 100)
training_features = features[0 : num_training_features]

# construct an array of size (451) for all features target values
outputcolumn = df.columns[4]
targets = df[outputcolumn].values
#select the first 85% as training tsrgets an array  size (383)
training_targets = targets[0 : num_training_features]

# layers include input layer + hidden layers + output layer
layers = [4, 9, 5, 1]
nn = MyNeuralNetwork(layers)

# call fit function with features (n_samples,n_features) and targets (n_samples)
nn.fit(training_features, training_targets)

print("L = ", nn.L, end="\n")
print("n = ", nn.n, end="\n")

print("xi = ", nn.xi, end="\n")
print("xi[0] = ", nn.xi[0], end="\n")
print("xi[1] = ", nn.xi[0], end="\n")

print("wh = ", nn.w, end="\n")
print("wh[1] = ", nn.w[1], end="\n")
