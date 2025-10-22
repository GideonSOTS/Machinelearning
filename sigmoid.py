# used for manipulating directory paths
import os
import numpy as np
from matplotlib import pyplot
from scipy import optimize
from plotData import plotData

# Load data
# The first two columns contains the exam scores and the third column
# contains the label.
data = np.loadtxt(os.path.join('Data', 'ex2data1.txt'), delimiter=',')
X, y = data[:, 0:2], data[:, 2]

plotData(X, y)
# add axes labels
pyplot.xlabel('Exam 1 score')
pyplot.ylabel('Exam 2 score')
pyplot.legend(['Admitted', 'Not admitted'])
pass
  

def sigmoid(z):
    g = np.zeros_like(z)
    g = 1 / (1 + np.exp(-z))
    return g

z = 0
g = sigmoid(z)
print('g(', z, ') = ', g)