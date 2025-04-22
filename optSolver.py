"""IOE 511/MATH 562, University of Michigan
Code template provided by: Albert S. Berahas & Jiahao Shi
Implemented by: Hugh Van Deventer
"""

import numpy as np

import algorithms
import functions


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
    print(f"Number of skipped steps ({method.name}): {n_skipped}")
    return x, f, history
