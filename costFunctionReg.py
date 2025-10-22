import os
import numpy as np
from scipy import optimize
import utils
from sigmoid import sigmoid 

def costFunctionReg(theta, X, y, lambda_):
    m = y.size  # number of training examples
    J = 0   # initialize cost
    h = sigmoid(X.dot(theta))   # hypothesis
    J = (-1/m) * (y.T.dot(np.log(h)) + (1 - y).T.dot(np.log(1 - h))) + (lambda_/(2*m)) * np.sum(theta[1:]**2)   # regularized cost function
    return J

# Load Data
# The first two columns contains the X values and the third column
# contains the label (y).
data = np.loadtxt(os.path.join('Data', 'ex2data2.txt'), delimiter=',')
X = data[:, :2]
y = data[:, 2]

