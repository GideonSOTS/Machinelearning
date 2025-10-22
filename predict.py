import os
import numpy as np
from sigmoid import sigmoid
from scipy import optimize
from costFunction import costFunction
from consFunction import gradFunction

def predict(theta, X):
    m = X.shape[0] # Number of training examples
    p = np.zeros(m) # Initialize predictions vector
    h = sigmoid(X.dot(theta))   # hypothesis
    p = (h >= 0.5).astype(int)  # Convert probabilities to 0 or 1 predictions
    return p

# Load data
data = np.loadtxt(os.path.join('Data', 'ex2data1.txt'), delimiter=',')
X, y = data[:, 0:2], data[:, 2]

m, n = X.shape  # Setup the data matrix appropriately, and add ones for the intercept term
X = np.concatenate([np.ones((m, 1)), X], axis=1)    # Add intercept term to X
print(X.shape)
# Initialize fitting parameters
initial_theta = np.zeros(n+1)

cost = costFunction(initial_theta, X, y)
grad = gradFunction(initial_theta, X, y)

print('Cost at initial theta (zeros): {:.3f}'.format(cost))
print('Expected cost (approx): 0.693\n')

print('Gradient at initial theta (zeros):')
print('\t[{:.4f}, {:.4f}, {:.4f}]'.format(*grad))
print('Expected gradients (approx):\n\t[-0.1000, -12.0092, -11.2628]\n')

# Compute and display cost and gradient with non-zero theta
test_theta = np.array([-24, 0.2, 0.2])
cost = costFunction(test_theta, X, y)
grad = gradFunction(test_theta, X, y)

print('Cost at test theta: {:.3f}'.format(cost))
print('Expected cost (approx): 0.218\n')

print('Gradient at test theta:')
print('\t[{:.3f}, {:.3f}, {:.3f}]'.format(*grad))
print('Expected gradients (approx):\n\t[0.043, 2.566, 2.647]')

# set options for optimize.minimize
options= {'maxiter': 400}

# Optimize
res = optimize.minimize(
    fun = lambda t: costFunction(t, X, y),
    x0 = initial_theta,
    jac = lambda t: gradFunction(t, X, y),
    method = 'TNC',
    options = options
)

# the optimized theta is in the x property
theta = res.x


# Print theta to screen
print('Cost at theta found by optimize.minimize: {:.3f}'.format(cost))
print('Expected cost (approx): 0.203\n');

print('theta:')
print('\t[{:.3f}, {:.3f}, {:.3f}]'.format(*theta))
print('Expected theta (approx):\n\t[-25.161, 0.206, 0.201]')

#  Predict probability for a student with score 45 on exam 1 
#  and score 85 on exam 2 
prob = sigmoid(np.dot([1, 45, 85], theta))
print('For a student with scores 45 and 85,'
      'we predict an admission probability of {:.3f}'.format(prob))
print('Expected value: 0.775 +/- 0.002\n')

# Compute accuracy on our training set
p = predict(theta, X)
print('Train Accuracy: {:.2f} %'.format(np.mean(p == y) * 100))
print('Expected accuracy (approx): 89.00 %')