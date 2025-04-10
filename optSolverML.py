import numpy as np

import algorithms
import functions


def optSolver(problem, method, options):
    """Function that runs a chosen ML algorithm on a chosen problem
    over a set of data points.

    Inputs:
        problem, method, options (structs)
    Outputs:
        final iterate (x) and final function value (f)
    """

    # Initialize history lists
    f_history = []

    # Get data
    X = problem.X


    # Compute initial function/gradient/Hessian
    x = problem.x0
    f = problem.compute_f(x)
    g = problem.compute_g(x)

    norm_g = np.linalg.norm(g, ord=np.inf)

    tol = max(1, norm_g)

    # Store initial values
    f_history.append(f)

    # set initial iteration counter
    k = 0
    n_skipped = 0
    # Theory and max_iters tolerance
    while norm_g > options.term_tol * tol and k < options.max_iterations:
        match method.name:
            case "GradientDescent":
                x_new, f_new, g_new, d, alpha = algorithms.GDStep(
                    x, f, g, problem, method, options
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
        ):
            H = H_new

        norm_g = np.linalg.norm(g, ord=np.inf)

        # increment iteration counter
        # print(k)
        # print(f"x: {x}")
        # print(f"f: {f}")
        # print(f"g: {g}")
        # if method.name == "Newton":
        #     print(f"H: {H}")
        # print(f"norm_g: {g}")

        # Store for plotting
        f_history.append(f)

        k = k + 1
    print(f"Number of skipped steps ({method.name}): {n_skipped}")
    return x, f  # , np.array(f_history)
