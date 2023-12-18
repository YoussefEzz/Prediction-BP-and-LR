import MyNeuralNetwork as NeuralNetwork
import pandas as pd
import numpy as np
import Utility as util

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
nn = NeuralNetwork.MyNeuralNetwork(layers,0.1, 0.9, util.linear)

# call fit function with features (n_samples,n_features) and targets (n_samples)
nn.fit(training_features, training_targets)

print("L = ", nn.L, end="\n")
print("n = ", nn.n, end="\n")

print("xi = ", nn.xi, end="\n")
print("xi[0] = ", nn.xi[0], end="\n")
print("xi[1] = ", nn.xi[0], end="\n")

print("wh = ", nn.w, end="\n")
print("wh[1] = ", nn.w[1], end="\n")