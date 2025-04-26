"""IOE 511/MATH 562, University of Michigan
Code template provided by: Albert S. Berahas & Jiahao Shi
Implemented by: Hugh Van Deventer
"""

import numpy as np

import algorithms


################## Continuous Optimization ##################


def optSolver(problem, method, options):
    """Function that runs a chosen algorithm on a chosen problem

    Inputs:
        problem, method, options (structs)
    Outputs:
        final iterate (x) and final function value (f)
    """

    # Initialize history lists
    history = {"iterations": [], "f": [], "norm_g": []}

    # Compute initial function/gradient/Hessian
    x = problem.x0
    f = problem.compute_f(x)
    g = problem.compute_g(x)

    # Handle Hessian cases
    if (
        method.name == "Newton"
        or method.name == "DFP"
        or method.name == "TRNewtonCG"
        or method.name == "TRSR1CG"
        or method.name == "ModifiedNewton"
    ):
        H = problem.compute_H(x)
    elif method.name == "BFGS":
        n = len(x)
        # Use custom H_0 if provided, otherwise use identity
        if "H_0" in method.options:
            H = method.options["H_0"]
        else:
            H = np.eye(n)
    elif method.name == "L-BFGS":
        from collections import deque

        n = len(x)
        # Use given memory size, otherwize use 6
        m = method.options.get("memory_size", 6)
        # Use deques for effecient maintence of memory
        s_list = deque(maxlen=m)
        y_list = deque(maxlen=m)

    # Trust Region parameters
    if method.name == "TRNewtonCG" or method.name == "TRSR1CG":
        if "Delta_0" in method.options:
            Delta = method.options["Delta_0"]
        else:
            Delta = 1.0

    norm_g = np.linalg.norm(g, ord=np.inf)

    tol = max(1, norm_g)

    # Store initial values
    history["iterations"].append(0)
    history["f"].append(f)
    history["norm_g"].append(norm_g)

    # set initial iteration counter
    k = 0
    n_skipped = 0
    # Theory and max_iters tolerance
    while norm_g > options.term_tol * tol and k < options.max_iterations:
        match method.name:
            case "GradientDescent":
                x_new, f_new, g_new = algorithms.GDStep(
                    x, f, g, problem, method, options
                )

            case "Newton":
                x_new, f_new, g_new, H_new = algorithms.NewtonStep(
                    x, f, g, H, problem, method, options
                )

            case "ModifiedNewton":
                x_new, f_new, g_new, H_new = algorithms.ModifiedNewtonStep(
                    x, f, g, H, problem, method, options
                )

            case "BFGS":
                x_new, f_new, g_new, H_new, n_skipped = algorithms.BFGSStep(
                    x, f, g, H, n_skipped, problem, method, options
                )

            case "L-BFGS":
                # s_list and y_list are updated in the L-BFGSStep function (pass by reference)
                x_new, f_new, g_new, n_skipped = algorithms.L_BFGSStep(
                    x, f, g, s_list, y_list, n_skipped, problem, method, options
                )

            case "DFP":
                x_new, f_new, g_new, H_new = algorithms.DFPStep(
                    x, f, g, H, problem, method, options
                )

            case "TRNewtonCG":
                x_new, f_new, g_new, H_new, Delta = algorithms.TRNewtonStep(
                    x, f, g, H, Delta, problem, method, options
                )

            case "TRSR1CG":
                x_new, f_new, g_new, H_new, Delta, n_skipped = algorithms.TRSR1Step(
                    x, f, g, H, Delta, n_skipped, problem, method, options
                )

            case _:
                raise ValueError("method is not implemented yet")

        # update old and new function values
        x_old = x
        f_old = f
        g_old = g

        if (
            method.name == "Newton"
            or method.name == "ModifiedNewton"
            or method.name == "BFGS"
            or method.name == "DFP"
            or method.name == "TRNewtonCG"
            or method.name == "TRSR1CG"
        ):
            H_old = H
        norm_g_old = norm_g

        x = x_new
        f = f_new
        g = g_new

        if (
            method.name == "Newton"
            or method.name == "ModifiedNewton"
            or method.name == "BFGS"
            or method.name == "DFP"
            or method.name == "TRNewtonCG"
            or method.name == "TRSR1CG"
        ):
            H = H_new

        norm_g = np.linalg.norm(g, ord=np.inf)

        # Store for plotting
        history["iterations"].append(k + 1)
        history["f"].append(f)
        history["norm_g"].append(norm_g)

        k = k + 1
    # print(f"Number of skipped steps ({method.name}): {n_skipped}")
    return x, f, history


################## ML Optimization ####################################
import algorithmsML

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


def optSolverML(problem, method, options):
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
            "grad_evals": [],
            "train_loss": [],
            "test_loss": [],
            "train_acc": [],
            "test_acc": [],
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
            batch_size = method.options["batch_size"]
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
    while k < 20 * n_train:
        # Compute function and gradient
        f, g = problem.compute_f_g(X_batch, y_batch, w)
        k += batch_size

        # Update weights
        w = algorithmsML.GDStep(X_batch, y_batch, w, f, g, k, problem, method, options)

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


################# Constrained Optimization ################################################

import algorithmsConst


def normGradLagrangian(x, lam, problem):
    """Function that computes the infinity norm of the gradient of the Lagrangian
    L(x, lam) = f(x) + lam^T * g(x)

    Inputs:
        x (current iterate), lam (Lagrange multipliers), problem, method, options (structs)
    Outputs:
        g (gradient of the Lagrangian)
    """
    # Evaluate the function and its gradient
    g = problem.compute_g(x)

    # Evaluate the constraints and their gradients
    J = problem.compute_const_g(x)

    # Compute the gradient of the Lagrangian
    g = g + np.dot(lam, J)  # Lagrangian gradient

    return np.linalg.norm(g, ord=np.inf)


def optSolverConst(problem, method, options):
    """Function that runs a chosen algorithm on a chosen constrained problem

    Inputs:
        problem, method, options (structs)
    Outputs:
        final iterate (x) and final function value (f)
    """

    # Initialize history lists
    history = {
        "iterations": [],
        "x": [],
        "f": [],
        "norm_c": [],
        "norm_g_lag": [],
        "nu": float,
        "avg_inner_iters": float,
    }

    # Initial function value and gradient
    x = problem.x0
    f = problem.compute_f(x)
    g = problem.compute_g(x)

    # Initial constraint value and gradient
    c = problem.compute_const(x)
    const_J = problem.compute_const_g(x)

    # Compute initial lagrange multipliers
    lamb = np.linalg.lstsq(const_J.T, -g)[0]

    # Initial penalty term
    nu = method.options.get("nu_0", 1e-4)

    # Penalty term update
    gamma = method.options.get("gamma", 10)

    # Termination condition terms
    norm_g_lag = normGradLagrangian(x, lamb, problem)
    lag_tol = max(1, norm_g_lag)
    norm_c = np.linalg.norm(c, ord=np.inf)
    const_tol = max(1, norm_c)
    epsilon = options.term_tol

    # Store initial values
    history["iterations"].append(0)
    history["x"].append(x)
    history["f"].append(f)
    history["norm_c"].append(norm_c)
    history["norm_g_lag"].append(norm_g_lag)
    history["nu"] = nu

    # Count number of inner iterations
    inner_iters = 0

    # set initial iteration counter
    k = 0
    # Theory and max_iters tolerance
    while (
        norm_g_lag > epsilon * lag_tol or norm_c > epsilon * const_tol
    ) and k < options.max_iterations:
        match method.name:
            case "QuadraticPenalty":
                x, f, num_inner_iters = algorithmsConst.QuadPenaltyStep(
                    x, nu, problem, method, options
                )
                # Compute new gradient and Jacobian
                g = problem.compute_g(x)
                const_J = problem.compute_const_g(x)
                # Compute new Lagrange multipliers
                lamb = np.linalg.lstsq(const_J.T, -g)[0]
                # Compute new constraint values
                c = problem.compute_const(x)
            case _:
                raise ValueError("method is not implemented yet")
        # Update termination conditions
        norm_g_lag = normGradLagrangian(x, lamb, problem)
        norm_c = np.linalg.norm(c, ord=np.inf)

        # Store for plotting
        history["iterations"].append(k + 1)
        history["x"].append(x)
        history["f"].append(f)
        history["norm_c"].append(norm_c)
        history["norm_g_lag"].append(norm_g_lag)
        history["nu"] = nu

        # Update inner iteration count
        inner_iters += num_inner_iters

        # Update penalty term
        nu = nu * gamma

        # Update iteration counter
        k = k + 1
    history["avg_inner_iters"] = inner_iters / k
    return x, f, norm_c, history
