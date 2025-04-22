# IOE 511/MATH 562, University of Michigan
# Code written by: Albert S. Berahas & Jiahao Shi

import numpy as np
import scipy.stats as stats
import scipy.sparse as sparse
import scipy.io

# Define all the functions and calculate their gradients and Hessians, those functions include:
# (1)(2)(3)(4) Quadractic function
# (5)(6) Quartic function
# (7)(8) Rosenbrock function
# (9) Data fit
# (10)(11) Exponential


# Problem Number: 1
# Problem Name: quad_10_10
# Problem Description: A randomly generated convex quadratic function; the
#                      random seed is set so that the results are
#                      reproducable. Dimension n = 10; Condition number
#                      kappa = 10


# function that computes the function value of the quad_10_10 function
def quad_10_10_func(x):
    # set raondom seed
    np.random.seed(0)
    # Generate random data
    q = np.random.normal(size=(10, 1))

    mat = scipy.io.loadmat("quad_10_10_Q.mat")
    Q = mat["Q"]

    # compute function value
    return (1 / 2 * x.T @ Q @ x + q.T @ x)[0]


def quad_10_10_grad(x):
    # set raondom seed
    np.random.seed(0)
    # Generate random data
    q = np.random.normal(size=(10, 1))
    mat = scipy.io.loadmat("quad_10_10_Q.mat")
    Q = mat["Q"]

    return Q @ x + q


def quad_10_10_Hess(x):
    # set raondom seed
    np.random.seed(0)
    # Generate random data
    q = np.random.normal(size=(10, 1))
    mat = scipy.io.loadmat("quad_10_10_Q.mat")
    Q = mat["Q"]

    return Q


# Problem Number: 2
# Problem Name: quad_10_1000
# Problem Description: A randomly generated convex quadratic function; the
#                      random seed is set so that the results are
#                      reproducable. Dimension n = 10; Condition number
#                      kappa = 1000

# function that computes the function value of the quad_10_1000 function


def quad_10_1000_func(x):
    # set raondom seed
    np.random.seed(0)
    # Generate random data
    q = np.random.normal(size=(10, 1))

    mat = scipy.io.loadmat("quad_10_1000_Q.mat")
    Q = mat["Q"]

    # compute function value
    return (1 / 2 * x.T @ Q @ x + q.T @ x)[0]


def quad_10_1000_grad(x):
    # set raondom seed
    np.random.seed(0)
    # Generate random data
    q = np.random.normal(size=(10, 1))

    mat = scipy.io.loadmat("quad_10_1000_Q.mat")
    Q = mat["Q"]

    return Q @ x + q


def quad_10_1000_Hess(x):
    # set raondom seed
    np.random.seed(0)
    # Generate random data
    q = np.random.normal(size=(10, 1))

    mat = scipy.io.loadmat("quad_10_1000_Q.mat")
    Q = mat["Q"]

    return Q


# Problem Number: 3
# Problem Name: quad_1000_10
# Problem Description: A randomly generated convex quadratic function; the
#                      random seed is set so that the results are
#                      reproducable. Dimension n = 1000; Condition number
#                      kappa = 10

# function that computes the function value of the quad_1000_10 function


def quad_1000_10_func(x):
    # set raondom seed
    np.random.seed(0)
    # Generate random data
    q = np.random.normal(size=(1000, 1))

    mat = scipy.io.loadmat("quad_1000_10_Q.mat")
    Q = mat["Q"]

    # compute function value
    return (1 / 2 * x.T @ Q @ x + q.T @ x)[0]


def quad_1000_10_grad(x):
    # set raondom seed
    np.random.seed(0)
    # Generate random data
    q = np.random.normal(size=(1000, 1))

    mat = scipy.io.loadmat("quad_1000_10_Q.mat")
    Q = mat["Q"]

    return Q @ x + q


def quad_1000_10_Hess(x):
    # set raondom seed
    np.random.seed(0)
    # Generate random data
    q = np.random.normal(size=(1000, 1))

    mat = scipy.io.loadmat("quad_1000_10_Q.mat")
    Q = mat["Q"]

    return Q


# Problem Number: 4
# Problem Name: quad_1000_1000
# Problem Description: A randomly generated convex quadratic function; the
#                      random seed is set so that the results are
#                      reproducable. Dimension n = 1000; Condition number
#                      kappa = 1000

# function that computes the function value of the quad_10_10 function


def quad_1000_1000_func(x):
    # set raondom seed
    np.random.seed(0)
    # Generate random data
    q = np.random.normal(size=(1000, 1))

    mat = scipy.io.loadmat("quad_1000_1000_Q.mat")
    Q = mat["Q"]

    # compute function value
    return (1 / 2 * x.T @ Q @ x + q.T @ x)[0]


def quad_1000_1000_grad(x):
    # set raondom seed
    np.random.seed(0)
    # Generate random data
    q = np.random.normal(size=(1000, 1))

    mat = scipy.io.loadmat("quad_1000_1000_Q.mat")
    Q = mat["Q"]

    return Q @ x + q


def quad_1000_1000_Hess(x):
    # set raondom seed
    np.random.seed(0)
    # Generate random data
    q = np.random.normal(size=(1000, 1))

    mat = scipy.io.loadmat("quad_1000_1000_Q.mat")
    Q = mat["Q"]

    return Q


# TODO: Finish 5
# Problem Number: 5
# Problem Name: quartic_1
# Problem Description: A quartic function. Dimension n = 4

# function that computes the function value of the quartic_1 function


def quartic_1_func(x):
    Q = np.array([[5, 1, 0, 0.5], [1, 4, 0.5, 0], [0, 0.5, 3, 0], [0.5, 0, 0, 2]])
    sigma = 1e-4

    return 1 / 2 * (x.T @ x) + sigma / 4 * (x.T @ Q @ x) ** 2


def quartic_1_grad(x):
    Q = np.array([[5, 1, 0, 0.5], [1, 4, 0.5, 0], [0, 0.5, 3, 0], [0.5, 0, 0, 2]])
    sigma = 1e-4

    return x + sigma * (x.T @ Q @ x) * (Q @ x)


def quartic_1_Hess(x):
    Q = np.array([[5, 1, 0, 0.5], [1, 4, 0.5, 0], [0, 0.5, 3, 0], [0.5, 0, 0, 2]])
    sigma = 1e-4

    return np.eye(4) + sigma * (2 * Q @ x) @ (Q @ x).T + sigma * (x.T @ Q @ x) * Q


# TODO: Finish 6 Derivative and Hessian

# Problem Number: 6
# Problem Name: quartic_2
# Problem Description: A quartic function. Dimension n = 4

# function that computes the function value of the quartic_2 function


def quartic_2_func(x):
    Q = np.array([[5, 1, 0, 0.5], [1, 4, 0.5, 0], [0, 0.5, 3, 0], [0.5, 0, 0, 2]])
    sigma = 1e4

    return 1 / 2 * (x.T @ x) + sigma / 4 * (x.T @ Q @ x) ** 2


def quartic_2_grad(x):
    Q = np.array([[5, 1, 0, 0.5], [1, 4, 0.5, 0], [0, 0.5, 3, 0], [0.5, 0, 0, 2]])
    sigma = 1e4

    return x + sigma * (x.T @ Q @ x) * (Q @ x)


def quartic_2_Hess(x):
    Q = np.array([[5, 1, 0, 0.5], [1, 4, 0.5, 0], [0, 0.5, 3, 0], [0.5, 0, 0, 2]])
    sigma = 1e4

    return np.eye(4) + sigma * (2 * Q @ x) @ (Q @ x).T + sigma * (x.T @ Q @ x) * Q


# TODO: Finish Rosenbrock function, gradient, and Hessian
# Problem Number: 7
# Problem Name: rosenbrock_2
# Problem Description: The Rosenbrock function. Dimension n = 2

# function that computes the function value of the rosenbrock_2 function


def rosenbrock_2_func(x):
    return 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2  # Rosenbrock function in 2D


def rosenbrock_2_grad(x):
    g = np.zeros(2)
    g[0] = -400 * x[0] * (x[1] - x[0] ** 2) + 2 * (x[0] - 1)
    g[1] = 200 * (x[1] - x[0] ** 2)
    return g


def rosenbrock_2_Hess(x):
    H = np.zeros((2, 2))
    H[0, 0] = -400 * x[1] + 1200 * x[0] ** 2 + 2
    H[0, 1] = -400 * x[0]
    H[1, 0] = -400 * x[0]
    H[1, 1] = 200
    return H


# Problem Number: 8
# Problem Name: rosenbrock_100
# Problem Description: The Rosenbrock function. Dimension n = 100


# function that computes the function value of the rosenbrock_100 function
def rosenbrock_100_func(x):
    f = 0
    for i in range(99):
        f = f + 100 * (x[i + 1] - x[i] ** 2) ** 2 + (1 - x[i]) ** 2
    return f


def rosenbrock_100_grad(x):
    g = np.zeros(100)
    for i in range(99):
        g[i] = -400 * x[i] * (x[i + 1] - x[i] ** 2) + 2 * (x[i] - 1)
        g[i + 1] += 200 * (x[i + 1] - x[i] ** 2)
    return g


def rosenbrock_100_Hess(x):
    H = np.zeros((100, 100))
    for i in range(99):
        H[i, i] += -400 * x[i + 1] + 1200 * x[i] ** 2 + 2
        H[i, i + 1] += -400 * x[i]
        H[i + 1, i] += -400 * x[i]
        H[i + 1, i + 1] += 200
    return H


# Problem Number: 9
# Problem Name: data_fit_2
# Problem Description: A data fit problem. Dimension n = 2


# function that computes the function value of the data_fit_2 function
def datafit_2_func(x):
    r"""Function that computes the function value for DataFit 2
        f(x) = sum_{i=1}^3 (y_i - x_0 * (1 - x_1^{i}))^2
    Input:
        x \in R^2
    Output:
        f(x)
    """
    y = np.array([1.5, 2.25, 2.625])
    f = 0
    for i in range(3):
        f += (y[i] - x[0] * (1 - x[1] ** (i + 1))) ** 2
    return f


def datafit_2_grad(x):
    r"""Function that computes the gradient of DataFit 2

    Input:
        x \in R^2
    Output:
        g = nabla f(x)
    """
    y = np.array([1.5, 2.25, 2.625])
    grad = np.zeros(2)
    for i in range(3):
        grad[0] += 2 * (y[i] - x[0] * (1 - x[1] ** (i + 1))) * (-1 + x[1] ** (i + 1))
        grad[1] += (
            2 * (y[i] - x[0] * (1 - x[1] ** (i + 1))) * x[0] * (i + 1) * x[1] ** i
        )
    return grad


def datafit_2_Hess(x):
    r"""Function that computes the Hessian of DataFit 2

    Input:
        x \in R^2
    Output:
        H = nabla^2 f(x)
    """
    y = np.array([1.5, 2.25, 2.625])
    H = np.zeros((2, 2))
    for i in range(3):
        H[0, 0] += 2 * (1 - x[1] ** (i + 1)) ** 2
        H[0, 1] += 2 * (
            -(1 - x[1] ** (i + 1)) * (x[0] * (i + 1) * x[1] ** i)
            + (y[i] - x[0] * (1 - x[1] ** (i + 1))) * (i + 1) * x[1] ** (i)
        )
        H[1, 0] += 2 * (
            -(1 - x[1] ** (i + 1)) * (x[0] * (i + 1) * x[1] ** i)
            + (y[i] - x[0] * (1 - x[1] ** (i + 1))) * (i + 1) * x[1] ** (i)
        )
        H[1, 1] += 2 * (
            (x[0] * (i + 1) * x[1] ** i) ** 2
            + (y[i] - x[0] * (1 - x[1] ** (i + 1)))
            * x[0]
            * (i + 1)
            * i
            * x[1] ** (i - 1)
        )
    return H


# Problem Number: 10
# Problem Name: exp_10
# Problem Description: An exponential function. Dimension n = 10


# function that computes the function value of the exp_10 function
def exponential_func(x):
    r"""Function that computes the function value for exponential function

    Input:
        x \in R^n
    Output:
        f(x)
    """
    f = ((np.exp(x[0]) - 1.0) / (np.exp(x[0]) + 1.0)) + 0.1 * np.exp(-x[0])
    for i in range(1, len(x)):
        f += (x[i] - 1.0) ** 4
    return f


def exponential_grad(x):
    r"""Function that computes the gradient of exponential function

    Input:
        x \in R^n
    Output:
        g = nabla f(x)
    """
    grad = np.zeros(len(x))
    grad[0] = 2 * np.exp(x[0]) / ((np.exp(x[0]) + 1.0) ** 2) - 0.1 * np.exp(-x[0])
    for i in range(1, len(x)):
        grad[i] = 4 * (x[i] - 1.0) ** 3
    return grad


def exponential_Hess(x):
    r"""Function that computes the Hessian of exponential function

    Input:
        x \in R^n
    Output:
        H = nabla^2 f(x)
    """
    H = np.zeros((len(x), len(x)))
    H[0, 0] = (2 * (np.exp(x[0]) - np.exp(2 * x[0]))) / (
        (np.exp(0) + 1.0) ** 3
    ) + 0.1 * np.exp(-x[0])
    for i in range(1, len(x)):
        H[i, i] = 12 * (x[i] - 1.0) ** 2
    return H


# Problem Number: 11
# Problem Name: exp_1000
# Problem Description: An exponential function. Dimension n = 1000

# function that computes the function value of the exp_1000 function


# Problem Number: 12
# Problem Name: genhumps_5
# Problem Description: A multi-dimensional problem with a lot of humps.
#                      This problem is from the well-known CUTEr test set.

# function that computes the function value of the genhumps_5 function


def genhumps_5_func(x):
    f = 0
    for i in range(4):
        f = (
            f
            + np.sin(2 * x[i]) ** 2 * np.sin(2 * x[i + 1]) ** 2
            + 0.05 * (x[i] ** 2 + x[i + 1] ** 2)
        )
    return f


# function that computes the gradient of the genhumps_5 function


def genhumps_5_grad(x):
    g = [
        4 * np.sin(2 * x[0]) * np.cos(2 * x[0]) * np.sin(2 * x[1]) ** 2 + 0.1 * x[0],
        4
        * np.sin(2 * x[1])
        * np.cos(2 * x[1])
        * (np.sin(2 * x[0]) ** 2 + np.sin(2 * x[2]) ** 2)
        + 0.2 * x[1],
        4
        * np.sin(2 * x[2])
        * np.cos(2 * x[2])
        * (np.sin(2 * x[1]) ** 2 + np.sin(2 * x[3]) ** 2)
        + 0.2 * x[2],
        4
        * np.sin(2 * x[3])
        * np.cos(2 * x[3])
        * (np.sin(2 * x[2]) ** 2 + np.sin(2 * x[4]) ** 2)
        + 0.2 * x[3],
        4 * np.sin(2 * x[4]) * np.cos(2 * x[4]) * np.sin(2 * x[3]) ** 2 + 0.1 * x[4],
    ]

    return np.array(g)


# function that computes the Hessian of the genhumps_5 function
def genhumps_5_Hess(x):
    H = np.zeros((5, 5))
    H[0, 0] = (
        8 * np.sin(2 * x[1]) ** 2 * (np.cos(2 * x[0]) ** 2 - np.sin(2 * x[0]) ** 2)
        + 0.1
    )
    H[0, 1] = (
        16 * np.sin(2 * x[0]) * np.cos(2 * x[0]) * np.sin(2 * x[1]) * np.cos(2 * x[1])
    )
    H[1, 1] = (
        8
        * (np.sin(2 * x[0]) ** 2 + np.sin(2 * x[2]) ** 2)
        * (np.cos(2 * x[1]) ** 2 - np.sin(2 * x[1]) ** 2)
        + 0.2
    )
    H[1, 2] = (
        16 * np.sin(2 * x[1]) * np.cos(2 * x[1]) * np.sin(2 * x[2]) * np.cos(2 * x[2])
    )
    H[2, 2] = (
        8
        * (np.sin(2 * x[1]) ** 2 + np.sin(2 * x[3]) ** 2)
        * (np.cos(2 * x[2]) ** 2 - np.sin(2 * x[2]) ** 2)
        + 0.2
    )
    H[2, 3] = (
        16 * np.sin(2 * x[2]) * np.cos(2 * x[2]) * np.sin(2 * x[3]) * np.cos(2 * x[3])
    )
    H[3, 3] = (
        8
        * (np.sin(2 * x[2]) ** 2 + np.sin(2 * x[4]) ** 2)
        * (np.cos(2 * x[3]) ** 2 - np.sin(2 * x[3]) ** 2)
        + 0.2
    )
    H[3, 4] = (
        16 * np.sin(2 * x[3]) * np.cos(2 * x[3]) * np.sin(2 * x[4]) * np.cos(2 * x[4])
    )
    H[4, 4] = (
        8 * np.sin(2 * x[3]) ** 2 * (np.cos(2 * x[4]) ** 2 - np.sin(2 * x[4]) ** 2)
        + 0.1
    )
    H[1, 0] = H[0, 1]
    H[2, 1] = H[1, 2]
    H[3, 2] = H[2, 3]
    H[4, 3] = H[3, 4]
    return H
