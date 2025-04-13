import numpy as np

import algorithmsML
import functionsML

record = True

def getBatch(X, y, batch_size):
    """Function that randomly selects a batch of data points
    from the dataset with replacement.

    Inputs:
        X, y (data points), batch_size (int)
    Outputs:
        X_batch, y_batch (selected data points)
    """
    n = len(X)
    if batch_size > n:
        raise ValueError("Batch size cannot be larger than the dataset size.")
    if batch_size == n:
        return X, y
    # Randomly select indices with replacement
    indices = np.random.choice(n, batch_size, replace=True)
    X_batch = X[indices]
    y_batch = y[indices]
    return X_batch, y_batch

def optSolver(problem, method, options):
    """Function that runs a chosen ML algorithm on a chosen problem
    over a set of data points.

    Inputs:
        problem, method, options (structs)
    Outputs:
        final iterate (x) and final function value (f)
    """

    # Initialize history lists
    if record:
        history = {
            'grad_evals': [],
            'train_loss': [],
            'test_loss': [],
            'train_acc': [],
            'test_acc': []
        }

    # Get data
    X_train = problem.X_train
    y_train = problem.y_train

    X_test = problem.X_test
    y_test = problem.y_test

    # Initialize weights
    w = problem.w0

    n_train = len(X_train)

    # Get initial batch size
    if method.name == "StochasticGradient":
        # Use given batch size, otherwise use 1
        if "batch_size" in method.options:
            batch_size = method.options['batch_size']
        else:
            batch_size = 1
    elif method.name == "GradientDescent":
        batch_size = n_train
    
    # Get initial batch
    if batch_size == n_train:
        X_batch = X_train
        y_batch = y_train
    elif batch_size < n_train:
        X_batch, y_batch = getBatch(X_train, y_train, batch_size)
    else:
        raise ValueError("Batch size cannot be larger than the dataset size.")


    # Record initial stats
    if record:
        history["grad_evals"].append(0)
        history["train_loss"].append(problem.compute_f(X_train, y_train, w))
        history["test_loss"].append(problem.compute_f(X_test, y_test, w))
        history["train_acc"].append(np.mean(problem.predict(X_train, w) == y_train))
        history["test_acc"].append(np.mean(problem.predict(X_test, w) == y_test))

    # set initial gradient evaluation counter
    k = 0
    # Theory and max_iters tolerance
    while k < 20*n_train:
        # Compute function and gradient
        f, g = problem.compute_f_g(X_batch, y_batch, w)
        k += batch_size

        # Update weights
        w = algorithmsML.GDStep(
            X_batch, y_batch, w, f, g, k, problem, method, options
        )

        match method.name:
            case "GradientDescent":
                pass
            case "StochasticGradient":
                # Update batch
                X_batch, y_batch = getBatch(X_train, y_train, batch_size)
            case _:
                raise ValueError("method is not implemented yet")

        # Store for plotting
        if record:
            # Record initial stats
            history["grad_evals"].append(k)
            history["train_loss"].append(problem.compute_f(X_train, y_train, w))
            history["test_loss"].append(problem.compute_f(X_test, y_test, w))
            history["train_acc"].append(np.mean(problem.predict(X_train, w) == y_train))
            history["test_acc"].append(np.mean(problem.predict(X_test, w) == y_test))

    # Compute final training loss
    train_loss = problem.compute_f(X_train, y_train, w)
    
    # Compute final training accuracy
    train_acc = np.mean(problem.predict(X_train, w) == y_train)

    # Compute final testing loss
    test_loss = problem.compute_f(X_test, y_test, w)

    # Compute final testing accuracy
    test_acc = np.mean(problem.predict(X_test, w) == y_test)

    # Return the final weights, final training loss, final training accuracy,
    #   final testing loss, final testing accuracy
    return w, train_loss, train_acc, test_loss, test_acc, history
