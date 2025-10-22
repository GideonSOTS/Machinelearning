import numpy as np
from sigmoid import sigmoid


def costFunction(theta, X, y):
   
    m = y.size  # number of training examples
    J = 0   # initialize cost
    h = sigmoid(X.dot(theta))   # hypothesis
    J = (-1/m) * (y.T.dot(np.log(h)) + (1 - y).T.dot(np.log(1 - h)))    # cost function
    
    return J