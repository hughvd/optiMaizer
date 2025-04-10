"""IOE 511/MATH 562, University of Michigan
Code template provided by: Albert S. Berahas & Jiahao Shi
Implemented by: Hugh Van Deventer


Define all the functions and calculate their gradients and Hessians, those functions include:
    (1) Rosenbrock function
    (2) Quadractic function
"""

import numpy as np


def rosen_func(x):
    """Function that computes the function value for the Rosenbrock function

    Input:
    Output:
        f(x)
    """

    return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2


def rosen_grad(x):
    """Function that computes the gradient of the Rosenbrock function

    Input:
        x
    Output:
        g = nabla f(x)
    """
    grad = np.zeros(2)
    grad[0] = -2 + 2 * x[0] - 400 * x[0] * x[1] + 400 * x[0] ** 3
    grad[1] = 200 * (x[1] - x[0] ** 2)
    return grad


def rosen_Hess(x):
    """Function that computes the Hessian of the Rosenbrock function

    Input:
        x
    Output:
        H = nabla^2 f(x)
    """
    H = np.zeros((2, 2))
    H[0, 0] = 2 - 400 * x[1] + 1200 * x[0] ** 2
    H[0, 1] = -400 * x[0]
    H[1, 0] = -400 * x[0]  # Hessian is symmetric
    H[1, 1] = 200
    return H


def quad_func(x, A, b, c):
    """Function that computes the function value for the Quadractic function

    Input:
        x
    Output:
        f(x)
    """
    return 0.5 * x.T @ A @ x + b.T @ x + c


def quad_grad(x, A, b, c):
    """Function that computes the gradient of the Quadratic function

    Input:
        x
    Output:
        g = nabla f(x)
    """
    return A @ x + b


def quad_Hess(x, A, b, c):
    """Function that computes the Hessian of the Quadratic function

    Input:
        x
    Output:
        H = nabla^2 f(x)
    """

    return A


def f2_func(x):
    """Function that computes the function value for function 2
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


def f2_grad(x):
    """Function that computes the gradient of function 2

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


def f2_Hess(x):
    """Function that computes the Hessian of function 2

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


def f3_func(x):
    """Function that computes the function value for function 3

    Input:
        x \in R^n
    Output:
        f(x)
    """
    f = ((np.exp(x[0]) - 1.0) / (np.exp(x[0]) + 1.0)) + 0.1 * np.exp(-x[0])
    for i in range(1, len(x)):
        f += (x[i] - 1.0) ** 4
    return f


def f3_grad(x):
    """Function that computes the gradient of function 3"

    Input:
        x \in R^n
    Output:
        g = nabla f(x)
    """
    grad = np.zeros(len(x))
    grad[0] = 2 * np.exp(x[0]) / ((np.exp(0) + 1.0) ** 2) - 0.1 * np.exp(-x[0])
    for i in range(1, len(x)):
        grad[i] = 4 * (x[i] - 1.0) ** 3
    return grad


def f3_Hess(x):
    """Function that computes the Hessian of function 3

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
