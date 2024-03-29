import numpy as np
from sklearn.model_selection import train_test_split

# Neural Network class
class MyNeuralNetwork:

  def __init__(self, layers, num_epochs, learning_rate, momentum, operation, validation_percent):
    self.L = len(layers)                                                              # number of layers
    self.n = layers.copy()                                                            # number of neurons in each layer
    self.eta = learning_rate                                                          # η learning rate
    self.alpha = momentum                                                             # α momentum
    self.validation_percent = validation_percent if validation_percent > 0 else None  # validation_percent
    self.num_epochs = num_epochs                                                      # epochs

    self.training_error = np.zeros((self.num_epochs, 2))                                                          # array of size (n_epochs, 2) that contain the evolution of the training error
    self.validation_error = np.zeros((self.num_epochs, 2))                                                          # array of size (n_epochs, 2) that contain the evolution of the validation error

    self.theta = [] #an array of arrays for the thresholds (θ)
    for lay in range(self.L):
      self.theta.append( np.random.random(layers[lay]))

    self.xi = []            # node values
    for lay in range(self.L):
      self.xi.append(np.zeros(layers[lay]))

    self.w = []             # edge weights
    self.w.append(np.zeros((1, 1)))
    for lay in range(1, self.L):
      self.w.append(np.zeros((layers[lay], layers[lay - 1])))

    self.h = []             # nan array of arrays for the fields (h)
    for lay in range(self.L):
      self.h.append(np.zeros(layers[lay]))

    self.delta = []         # an array of arrays for the propagation of errors (Δ)
    for lay in range(self.L):
      self.delta.append(np.zeros(layers[lay]))

    self.d_w = []           # an array of matrices for the changes of the weights (δw)
    self.d_w.append(np.zeros((1, 1)))
    for lay in range(1, self.L):
      self.d_w.append(np.zeros((layers[lay], layers[lay - 1])))

    self.d_w_prev = []      # an array of matrices for the previous changes of the weights, used for the momentum term (δw(prev))
    self.d_w_prev.append(np.zeros((1, 1)))
    for lay in range(1, self.L):
      self.d_w_prev.append(np.zeros((layers[lay], layers[lay - 1])))

    self.d_theta = []       # an array of arrays for the changes of the weights (δθ)
    for lay in range(self.L):
      self.d_theta.append(np.zeros(layers[lay]))

    self.d_theta_prev = []       # an array of arrays for the previous changes of the thresholds, used for the momentum term (δθ(prev))
    for lay in range(self.L):
      self.d_theta_prev.append(np.zeros(layers[lay]))

    self.fact = operation

  # initialize weights with random values
  def initialize_weights(self):
    self.w = []             # edge weights
    self.w.append(np.random.random((1, 1)))
    for lay in range(1, self.L):
      self.w.append(np.random.random((self.n[lay], self.n[lay - 1])))
    return
  
  # initialize weights with ones for test
  def initialize_weights_for_test(self):
    self.w = []             # edge weights
    self.w.append(np.ones((1, 1)))
    for lay in range(1, self.L):
      self.w.append(np.ones((self.n[lay], self.n[lay - 1])))
    return
  
  # initialize thresholds with zeros for test
  def initialize_thresholds_for_test(self):
    self.theta = [] #an array of arrays for the thresholds (θ)
    for lay in range(self.L):
      self.theta.append( np.zeros(self.n[lay]))
    return

  # X : an array of arrays size (n_samples,n_features), which holds the training samples represented as floating point
  #feature vectors; and a vector y of size (n_samples), which holds the target values for the training samples
  def fit(self, X, y):
    #split X and y into training and validation sets
    X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size = self.validation_percent, random_state= 42)


    num_training_patterns = X_train.shape[0]

    for epoch in range(0, self.num_epochs):
      
      for i in range(0 , num_training_patterns):
        
        #Feed−forward propagation of pattern xµ to obtain the output o(xµ)
        o = self.feedforward(X_train[i])
        #print("training sample : ", i, " ,x : ", X_train[i])
        #print(self.w)

        #Back−propagate the error for this pattern
        self.backpropagate(o, y_train[i])

        #update the weights
        self.update_weights()
       

          
        # print("sample : ", i, " ,x : ", X_train[i])
        # print(self.w)
      print("epoch = ", epoch)
      
      #Feed−forward all training patterns and calculate their prediction quadratic error
      self.training_error_compute(X_train, y_train, epoch)

      #Feed−forward all validation patterns and calculate their prediction quadratic error
      self.validation_error_compute(X_validation, y_validation, epoch)

    return
  
  #Feed−forward propagation of pattern xµ to obtain the output o(xµ) using vector multiplications
  def feedforward(self, x):
    self.h[0] = x
    self.xi[0] = x
    for lay in range(1, self.L):
      #transpose x_i to nl * 1
      xi_t = self.xi[lay - 1]
      #print(xi_t)

      # #transpose theta_i to nl * 1                                   
      theta_t = self.theta[lay]
      #print(theta_t)

      #multiply matrix w of size (nl * nl-1) * x_i of size (nl-1 * 1) + theta_i of size(nl * 1) to get h of size(nl * 1)                               
      self.h[lay] = np.dot(self.w[lay], xi_t) + theta_t
      #print(self.h[lay])

      #calculate fact of h of size(nl * 1) to get g of size(nl * 1) then assign it to x_i of size (nl * 1)
      self.xi[lay] = self.fact.g(self.h[lay])

    o = self.xi[self.L - 1]  
    return o
  
  #Back−propagate the error for each pattern
  def backpropagate(self, o, z):

    #Δ(L) = g'(h(L))(o - z)
    delta = (o - z)
    t1 = self.fact.g_diff(self.h[self.L - 1])
    t2 = t1 * delta
    self.delta[self.L - 1] = self.fact.g_diff(self.h[self.L - 1]) * (o - z)
    l = self.L - 1
    for j in range(l - 1, 0, -1):

      t3 = np.dot(self.delta[j + 1], self.w[j + 1])
      #print(type(t3))

      t4 = self.fact.g_diff(self.h[j])
      #print(t4)

      #print(t3 * t4) 
      self.delta[j] = self.fact.g_diff(self.h[j]) * np.dot(self.delta[j + 1], self.w[j + 1])
    return

  #Update weights and thresholds
  def update_weights(self):

    for lay in range(1, self.L):
      
      product_t = np.outer(self.delta[lay], self.xi[lay - 1])
      # print(product_t)
      # print(-1 * self.eta *product_t)
      # Amount of weights update
      # The elements of the resulting matrix are obtained by outer function by multiplying each element of vector1 by each element of vector2. The resulting matrix has dimensions len(vector1) x len(vector2)
      self.d_w[lay] = -1 * self.eta * np.outer(self.delta[lay], self.xi[lay - 1]) + self.alpha * self.d_w_prev[lay]
      # print("self.delta[",lay,"] : ", self.delta[lay])
      # print("self.xi[", lay - 1, "] : ", self.xi[lay - 1])
      # print("self.d_w[", lay, "] : ", self.d_w[lay])
      # Amount of thresholds update
      self.d_theta[lay]  = self.eta * self.delta[lay] + self.alpha * self.d_theta_prev[lay]


      # update all the weights and thresholds changes we applied in the previous step
      self.d_w_prev[lay]     = self.d_w[lay]

      self.d_theta_prev[lay] = self.d_theta[lay]

      # Finally, update all the weights and thresholds
      self.w[lay] = self.w[lay] + self.d_w[lay]
      # print(self.w)
      self.theta[lay] = self.theta[lay] + self.d_theta[lay]

    return 
  
  #trainig error compute
  #PQE = Σ(y_pred - y_actual)^2 / n
  def training_error_compute(self, X_train, y_train, epoch):
    num_patterns = X_train.shape[0]
    PQE = 0
    for i in range(0 , num_patterns):

        z = y_train[i]
        #Feed−forward propagation of pattern xµ to obtain the output o(xµ)
        o = self.feedforward(X_train[i])
        PQE = PQE + ((o - z) ** 2)

    PQE = PQE / num_patterns
    self.training_error[epoch, 0] = epoch
    self.training_error[epoch, 1] = PQE
    

  #validation error compute
  #PQE = Σ(y_pred - y_actual)^2 / n
  def validation_error_compute(self, X_validation, y_validation, epoch):
    num_patterns = X_validation.shape[0]
    PQE = 0
    for i in range(0 , num_patterns):
   
        z = y_validation[i]
        #Feed−forward propagation of pattern xµ to obtain the output o(xµ)
        o = self.feedforward(X_validation[i])
        PQE = PQE + ((o - z) ** 2)

    PQE = PQE / num_patterns
    self.validation_error[epoch, 0] = epoch
    self.validation_error[epoch, 1] = PQE

  #
  def loss_epochs(self):
    return self.training_error, self.validation_error