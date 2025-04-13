# Machine learning loss functions
import numpy as np

# Linear Least Squares

def linear_least_squares_predict(X, w):
    """Predict class labels using linear least squares.

    Args:
        X: Input data matrix (n_samples, n_features).
        w: Weights (n_features,).

    Returns:
        predictions: Predicted class labels (n_samples,).
    """
    return np.sign(X @ w)

def linear_least_squares_func(X, y, w):
    """Compute the linear least squares loss.

    Args:
        X: Input data matrix (n_samples, n_features).
        y: Target values (n_samples,).
        w: Weights (n_features,).

    Returns:
        loss: The linear least squares loss.
    """
    residuals = X @ w - y
    loss = 0.5 * np.mean(residuals ** 2)
    return loss

def linear_least_squares_grad(X, y, w):
    """Compute the linear least squares gradient.

    Args:
        X: Input data matrix (n_samples, n_features).
        y: Target values (n_samples,).
        w: Weights (n_features,).

    Returns:
        gradient: The gradient of the loss with respect to w.
    """
    residuals = X @ w - y
    gradient = X.T @ residuals / len(y)
    return gradient

def linear_least_squares_func_grad(X, y, w):
    """Compute the linear least squares loss and gradient.

    Args:
        X: Input data matrix (n_samples, n_features).
        y: Target values (n_samples,).
        w: Weights (n_features,).

    Returns:
        loss: The linear least squares loss.
        gradient: The gradient of the loss with respect to w.
    """
    residuals = X @ w - y
    loss = 0.5 * np.mean(residuals ** 2)
    gradient = X.T @ residuals / len(y)
    return loss, gradient


# Logistic Regression

def logistic_regression_predict(X, w):
    """Predict class labels using logistic regression.

    Args:
        X: Input data matrix (n_samples, n_features).
        w: Weights (n_features,).

    Returns:
        predictions: Predicted class labels (n_samples,).
    """
    logits = X @ w
    probabilities = 1 / (1 + np.exp(-logits))
    return np.where(probabilities >= 0.5, 1, -1)

def logistic_regression_func(X, y, w):
    """Compute the logistic regression loss.

    Args:
        X: Input data matrix (n_samples, n_features).
        y: Target values (n_samples,).
        w: Weights (n_features,).

    Returns:
        loss: The logistic regression loss.
    """
    logits = X @ w
    # Use logaddexp for numerical stability
    loss = np.mean(np.logaddexp(0, -y * logits))
    return loss


def logistic_regression_grad(X, y, w):
    """Compute the logistic regression gradient.

    Args:
        X: Input data matrix (n_samples, n_features).
        y: Target values (n_samples,).
        w: Weights (n_features,).

    Returns:
        gradient: The gradient of the loss with respect to w.
    """
    logits = X @ w
    sigmoid_term = -y / (1 + np.exp(y * logits))
    gradient = X.T @ sigmoid_term / len(y)

    return gradient


def logistic_regression_func_grad(X, y, w):
    """Compute the logistic regression loss and gradient.

    Args:
        X: Input data matrix (n_samples, n_features).
        y: Target values (n_samples,).
        w: Weights (n_features,).

    Returns:
        loss: The logistic regression loss.
        gradient: The gradient of the loss with respect to w.
    """
    logits = X @ w
    # Use logaddexp for numerical stability
    loss = np.mean(np.logaddexp(0, -y * logits))
    
    # Avoid direct calculation of exp(-y*logits)
    sigmoid_term = -y / (1 + np.exp(y * logits))
    gradient = X.T @ sigmoid_term / len(y)

    return loss, gradient