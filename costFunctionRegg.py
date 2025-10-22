import os
import numpy as np
from scipy import optimize
from matplotlib import pyplot
import utils
from utils import mapFeature
from sigmoid import sigmoid
from costFunctionReg import costFunctionReg
from predict import predict

def gradFunctionReg(theta, X, y, lambda_):
    m = y.size                      # number of training examples
    grad = np.zeros(theta.shape)    # gradient
    h = sigmoid(X.dot(theta))       # hypothesis
    grad[0] = (1/m) * (X[:, 0].T.dot(h - y))                               # gradient for theta_0
    grad[1:] = (1/m) * (X[:, 1:].T.dot(h - y)) + (lambda_/m) * theta[1:]   # gradient for theta_j where j >= 1
    return grad

# Load Data
# The first two columns contains the X values and the third column
# contains the label (y).
data = np.loadtxt(os.path.join('Data', 'ex2data2.txt'), delimiter=',')
X = data[:, :2]
y = data[:, 2]

X = mapFeature(X[:, 0], X[:, 1])
# Initialize fitting parameters
initial_theta = np.zeros(X.shape[1])

# Set regularization parameter lambda to 1
# DO NOT use `lambda` as a variable name in python
# because it is a python keyword
lambda_ = 1

# Compute and display initial cost and gradient for regularized logistic
# regression
cost = costFunctionReg(initial_theta, X, y, lambda_)
grad = gradFunctionReg(initial_theta, X, y, lambda_)


print('Cost at initial theta (zeros): {:.3f}'.format(cost))
print('Expected cost (approx)       : 0.693\n')

print('Gradient at initial theta (zeros) - first five values only:')
print('\t[', ', '.join('{:.4f}'.format(g) for g in grad), ']')
print('Expected gradients (approx) - first five values only:')
print('\t[0.0085, 0.0188, 0.0001, 0.0503, 0.0115]\n')


# Compute and display cost and gradient
# with all-ones theta and lambda = 10
test_theta = np.ones(X.shape[1])
cost = costFunctionReg(test_theta, X, y, 10)
grad = gradFunctionReg(test_theta, X, y, 10)

print('------------------------------\n')
print('Cost at test theta    : {:.2f}'.format(cost))
print('Expected cost (approx): 3.16\n')

print('Gradient at test theta - first five values only:')
print('\t[', ', '.join('{:.4f}'.format(g) for g in grad), ']')
#print('\t[{:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}]'.format(*grad[:5]))
print('Expected gradients (approx) - first five values only:')
print('\t[0.3460, 0.1614, 0.1948, 0.2269, 0.0922]')



# Initialize fitting parameters
initial_theta = np.zeros(X.shape[1])

# Set regularization parameter lambda to 1 (you should vary this)
lambda_ = 1

# set options for optimize.minimize
options= {'maxfun': 100}

# Optimize
res = optimize.minimize(
    fun = lambda t: costFunctionReg(t, X, y, lambda_),
    x0 = initial_theta,
    jac = lambda t: gradFunctionReg(t, X, y, lambda_),
    method = 'TNC',
    options = options
)

# the fun property of OptimizeResult object returns
# the value of costFunction at optimized theta
cost = res.fun

# the optimized theta is in the x property of the result
theta = res.x

utils.plotDecisionBoundary(utils.plotData, theta, X, y)
pyplot.xlabel('Microchip Test 1')
pyplot.ylabel('Microchip Test 2')
pyplot.legend(['y = 1', 'y = 0'])
pyplot.grid(False)
pyplot.title('lambda = %0.2f' % lambda_)

# Compute accuracy on our training set
p = predict(theta, X)

print('Train Accuracy: %.1f %%' % (np.mean(p == y) * 100))
print('Expected accuracy (with lambda = 1): 83.1 % (approx)\n')
