import numpy as np
import matplotlib.pyplot as plt

def plotData(X, y):
    """
    Plot the data points X and y.
    Assumes X has at least two columns.
    """
    pos = y == 1
    neg = y == 0
    plt.scatter(X[pos, 0], X[pos, 1], c='k', marker='+', label='y = 1')
    plt.scatter(X[neg, 0], X[neg, 1], c='y', marker='o', edgecolors='k', label='y = 0')
    plt.legend()

def plotDecisionBoundary(plotData, theta, X, y):
    """
    Plots data points with the decision boundary defined by theta.
    Works for both simple and mapped features.
    """
    if X.shape[1] <= 3:
        # 2D data (linear decision boundary)
        plotData(X[:, 1:3], y)
        x_min, x_max = X[:, 1].min(), X[:, 1].max()
        plot_x = np.array([x_min, x_max])
        plot_y = (-1 / theta[2]) * (theta[1] * plot_x + theta[0])
        plt.plot(plot_x, plot_y, 'b-')
    else:
        # Higher-order feature mapping (nonlinear boundary)
        plotData(X[:, 1:3], y)

        # Create grid
        u = np.linspace(-1, 1.5, 50)
        v = np.linspace(-1, 1.5, 50)
        z = np.zeros((u.size, v.size))

        for i in range(len(u)):
            for j in range(len(v)):
                z[i, j] = np.dot(mapFeature(u[i], v[j]), theta)
        z = z.T

        plt.contour(u, v, z, levels=[0], colors='g', linewidths=2)
        plt.contourf(u, v, z, levels=[np.min(z), 0, np.max(z)], cmap='Greens', alpha=0.3)
import numpy as np

import numpy as np

def mapFeature(x1, x2, degree=6):
    """
    Feature mapping function to polynomial features.
    Maps the two input features to polynomial features up to the given degree.
    Returns a new feature array with more features, comprising
    X1, X2, X1**2, X2**2, X1*X2, X1*X2**2, etc.
    """

    # Ensure x1, x2 are numpy arrays
    x1 = np.array(x1)
    x2 = np.array(x2)

    if x1.ndim == 0:  # i.e. scalar
        x1 = np.array([x1])
        x2 = np.array([x2])

    out = [np.ones(x1.shape[0])]

    for i in range(1, degree + 1):
        for j in range(i + 1):
            out.append((x1 ** (i - j)) * (x2 ** j))

    return np.stack(out, axis=1)

