import numpy as np
from scipy import optimize
from costFunction import costFunction
from sigmoid import sigmoid


def gradFunction(theta, X, y):
   
    m = y.size  # number of training examples
    grad = np.zeros(theta.shape)    # initialize gradient
    h = sigmoid(X.dot(theta))   # hypothesis
    grad = (1/m) * (X.T.dot(h - y))     # gradient
    
    return grad