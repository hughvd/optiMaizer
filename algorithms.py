"""IOE 511/MATH 562, University of Michigan
Code template provided by: Albert S. Berahas & Jiahao Shi
Implemented by: Hugh Van Deventer
"""

import numpy as np
import scipy

# Helpers


def compute_step_size(x, d, f, g, problem, method):
    """Computes step size based on specified method"""
    match method.options["step_type"]:
        case "Constant":
            alpha = method.options["alpha"]
        case "Backtracking":
            # Get params
            alpha = method.options["alpha"]
            tau = method.options["tau"]
            c1 = method.options["c_1_ls"]

            # Initial step
            x_new = x + alpha * d
            f_new = problem.compute_f(x_new)

            # Backtracking
            while f_new > f + c1 * alpha * g @ d:
                alpha = tau * alpha
                x_new = x + alpha * d
                f_new = problem.compute_f(x_new)

                # Break if step size becomes too small
                if alpha < 1e-10:
                    break
        case "Wolfe":
            # Strong wolfe line search
            # Line search parameters
            c1 = method.options["c_1_ls"]
            c2 = method.options["c_2_ls"]

            # Subroutine parameters
            if "alpha_low" in method.options:
                alpha_low = method.options["alpha_low"]
            else:
                alpha_low = 0
            if "alpha_high" in method.options:
                alpha_high = method.options["alpha_high"]
            else:
                alpha_high = 1000
            if "alpha" in method.options:
                alpha = method.options["alpha"]
            else:
                alpha_ = 1
            if "c" in method.options:
                c = method.options["c"]
            else:
                c = 0.5

            while True:
                x_new = x + alpha * d
                # Sufficient decrease condition
                if problem.compute_f(x_new) <= f + c1 * alpha * g.T @ d:
                    # Curvature condition
                    if problem.compute_g(x_new).T @ d >= c2 * g.T @ d:
                        break
                    else:
                        alpha_l = alpha
                else:
                    alpha_h = alpha
                alpha = c * alpha_l + (1 - c) * alpha_h

        case _:
            raise ValueError("step type is not defined")

    return alpha


def ConjugateGradientSubproblem(f, g, H, problem, method):
    """Computes the conjugate gradient subproblem"""
    # Initialize variables
    z = np.zeros_like(g)
    r = g.copy()
    p = -g.copy()
    B = H.copy()

    # Initial check
    if np.sqrt(r.T @ r) < method.options["term_tol_CG"]:
        return z

    for _ in range(len(g)):

        # Concave case
        if p.T @ B @ p <= 0:
            # TODO: Find τ such that dk = zj + τ pj minimizes mk (dk ) and satisfies ‖dk ‖ = ∆k
            return

        alpha = (r.T @ r) / (p.T @ B @ p)

        z_new = z + alpha * p

        # If outside of region
        if np.sqrt(z_new.T @ z_new) >= method.options["region_size"]:
            # TODO: Find τ such that dk = zj + τ pj minimizes mk (dk ) and satisfies ‖dk ‖ = ∆k
            return

        r_new += alpha * B @ p

        if np.sqrt(r_new.T @ r_new) <= method.options["term_tol_CG"]:
            return z_new

        beta = (r_new.T @ r_new) / (r.T @ r)
        p = -r_new + beta * p

        # Setting variables for next ieration
        r = r_new
        z = z_new

    return z


# Descent methods


def GDStep(x, f, g, problem, method, options):
    """Function that: (1) computes the GD step; (2) updates the iterate; and,
         (3) computes the function and gradient at the new iterate

    Inputs:
        x, f, g, problem, method, options
    Outputs:
        x_new, f_new, g_new, d, alpha
    """
    # Set the search direction d to be -g
    d = -g

    # Compute step size
    alpha = compute_step_size(x, d, f, g, problem, method)

    # Update
    x_new = x + alpha * d
    f_new = problem.compute_f(x_new)
    g_new = problem.compute_g(x_new)

    return x_new, f_new, g_new, d, alpha


def NewtonStep(x, f, g, H, problem, method, options):
    """Function that: (1) computes the Newton step; (2) updates the iterate; and,
        (3) computes the function and gradient at the new iterate

    Inputs:
        x, f, g, H, problem, method, options
    Outputs:
        x_new, f_new, g_new, H_new, d, alpha
    """
    # Compute Newton direction by solving Hd = -g
    try:
        d = np.linalg.solve(H, -g)
    except np.linalg.LinAlgError:
        # If Hessian is singular, fall back to gradient descent
        d = -g

    # Compute step size
    alpha = compute_step_size(x, d, f, g, problem, method)

    # Update
    x_new = x + alpha * d
    f_new = problem.compute_f(x_new)
    g_new = problem.compute_g(x_new)
    H_new = problem.compute_H(x_new)

    return x_new, f_new, g_new, H_new, d, alpha


def ModifiedNewtonStep(x, f, g, H, problem, method, options):
    """Function that: (1) computes the modified Newton step; (2) updates the iterate; and,
        (3) computes the function and gradient at the new iterate

    Inputs:
        x, f, g, H, problem, method, options
    Outputs:
        x_new, f_new, g_new, H_new, d, alpha
    """
    # Compute PSD Hessian
    n = H.shape[0]
    beta = method.options["beta"]

    # Initialize eta
    min_diag = np.min(np.diag(H))
    if min_diag > 0:
        eta = 0
    else:
        eta = -min_diag + beta

    # Cholesky Factorization
    max_iterations = 1000
    success = False
    for k in range(max_iterations):
        # Attempt Cholesky factorization of H + eta*I
        try:
            # Add eta*I to H
            H_mod = H.copy()
            H_mod[np.diag_indices(n)] += eta

            # Cholesky factorization
            L = scipy.linalg.cholesky(H_mod, lower=True, check_finite=False)
            if eta != 0:
                print("Modified Newton: Cholesky factorization successful")
            # If successful, break
            success = True
            break

        except scipy.linalg.LinAlgError:
            # Step 8: Increase eta
            eta = max(2 * eta, beta)

    # Check if we successfully computed the cholesky factorization
    if not success:
        raise RuntimeError(
            f"Failed to compute Cholesky factorization after {max_iterations} iterations"
        )

    # Compute the Newton direction using LL^T with forward/backward subsitution
    # Solve L*y = -g for y
    y = scipy.linalg.solve_triangular(L, -g, lower=True)
    # Solve L.T*d = y for d
    d = scipy.linalg.solve_triangular(L.T, y, lower=False)

    # Compute step size
    alpha = compute_step_size(x, d, f, g, problem, method)

    # Update
    x_new = x + alpha * d
    f_new = problem.compute_f(x_new)
    g_new = problem.compute_g(x_new)
    H_new = problem.compute_H(x_new)

    return x_new, f_new, g_new, H_new, d, alpha


def BFGSStep(x, f, g, H, n_skipped, problem, method, options):
    """Function that: (1) computes the BFGS step; (2) updates the iterate; and,
        (3) computes the function and gradient at the new iterate
    *Note: This function uses the BFGS formula to update the inverse Hessian H
    Inputs:
        x, f, g, H, problem, method, options
    Outputs:
        x_new, f_new, g_new, H_new, d, alpha
    """
    # Compute search direction (H is now the inverse Hessian)
    d = -H @ g

    # Compute step size
    alpha = compute_step_size(x, d, f, g, problem, method)

    # Update
    x_new = x + alpha * d
    f_new = problem.compute_f(x_new)
    g_new = problem.compute_g(x_new)

    # Update Hessian with BFGS formula
    s = x_new - x
    y = g_new - g
    if s @ y > method.options["epsilon_sy"] * np.linalg.norm(s) * np.linalg.norm(y):
        rho = 1 / (s @ y)
        I = np.eye(len(x))
        H_new = (I - rho * np.outer(s, y)) @ H @ (
            I - rho * np.outer(y, s)
        ) + rho * np.outer(s, s)
    else:
        H_new = H
        n_skipped += 1

    return x_new, f_new, g_new, H_new, d, alpha, n_skipped


def L_BFGSStep(x, f, g, s_list, y_list, n_skipped, problem, method, options):
    """Function that: (1) computes the L-BFGS step; (2) updates the iterate; and,
        (3) computes the function and gradient at the new iterate

    Inputs:
        x, f, g, H, problem, method, options
    Outputs:
        x_new, f_new, g_new, H_new, d, alpha
    """
    # L-BFGS update
    q = g.copy()
    alpha = np.zeros(len(s_list))

    # First loop - most recent to oldest (backward)
    for i in range(len(s_list) - 1, -1, -1):
        rho_i = 1.0 / (s_list[i] @ y_list[i])
        alpha[i] = rho_i * (s_list[i] @ q)
        q -= alpha[i] * y_list[i]

    # Initial Hessian approximation (identity matrix)
    r = q

    # Second loop - oldest to most recent (forward)
    for i in range(len(s_list)):
        rho_i = 1.0 / (s_list[i] @ y_list[i])
        beta = rho_i * (y_list[i] @ r)
        r += s_list[i] * (alpha[i] - beta)

    d = -r

    # Compute step size
    alpha = compute_step_size(x, d, f, g, problem, method)

    # Update
    x_new = x + alpha * d
    f_new = problem.compute_f(x_new)
    g_new = problem.compute_g(x_new)

    # Update memory (When we are at capacity, deque will automatically remove the oldest entry)
    s = x_new - x
    y = g_new - g

    if s @ y >= method.options["epsilon_sy"] * np.linalg.norm(s) * np.linalg.norm(y):
        s_list.append(s)
        y_list.append(y)
    else:
        n_skipped += 1

    return x_new, f_new, g_new, d, alpha, n_skipped


# DFP
def DFPStep(x, f, g, H, problem, method, options):
    """Function that: (1) computes the DFP step; (2) updates the iterate; and,
        (3) computes the function and gradient at the new iterate

    Inputs:
        x, f, g, H, problem, method, options
    Outputs:
        x_new, f_new, g_new, H_new, d, alpha
    """
    # Compute search direction (H is now the inverse Hessian)
    d = -H @ g

    # Compute step size
    alpha = compute_step_size(x, d, f, g, problem, method)

    # Update
    x_new = x + alpha * d
    f_new = problem.compute_f(x_new)
    g_new = problem.compute_g(x_new)

    # Update Hessian with DFP formula
    s = x_new - x
    y = g_new - g
    if s @ y > method.options["epsilon_sy"] * np.linalg.norm(s) * np.linalg.norm(y):
        rho = 1 / (s @ y)
        H_new = H + (rho * np.outer(s, s)) - (H @ np.outer(y, y) @ H / np.dot(y, H @ y))
    else:
        H_new = H

    return x_new, f_new, g_new, H_new, d, alpha


# Trust region methods


# TRNewtonCG
def TRNewtonStep(x, f, g, H, problem, method, options):
    """Function that: (1) computes the TR Newton step; (2) updates the iterate; and,
        (3) computes the function and gradient at the new iterate

    Inputs:
        x, f, g, H, problem, method, options
    Outputs:
        x_new, f_new, g_new, H_new, d, alpha
    """
    # Compute Newton direction by solving Hd = -g
    try:
        d = np.linalg.solve(H, -g)
    except np.linalg.LinAlgError:
        # If Hessian is singular, fall back to gradient descent
        d = -g

    # Solve TR subproblem
    d = ConjugateGradientSubproblem(f, g, H, problem, method)

    # Compute actual vs prediction reduction ratio
    rho = (f - problem.compute_f(x + d)) / (f - (f + g @ d + 0.5 * d.T @ H @ d))

    # Check if the step is acceptable
    if rho > method.options["c_1_tr"]:
        # Accept the step
        x_new = x + d
        f_new = problem.compute_f(x_new)
        g_new = problem.compute_g(x_new)
        H_new = problem.compute_H(x_new)

        # Update trust region radius
        if rho > method.options["c_2_tr"]:
            Delta = 2 * Delta

        return x_new, f_new, g_new, H_new, d, Delta
    else:
        # Reject the step and reduce the trust region radius
        Delta = 0.5 * Delta

    return x, f, g, H, d, Delta


# TRSR1CG
def TRSR1Step(x, f, g, s_list, y_list, n_skipped, problem, method, options):
    """Function that: (1) computes the TR Newton step; (2) updates the iterate; and,
        (3) computes the function and gradient at the new iterate

    Inputs:
        x, f, g, H, problem, method, options
    Outputs:
        x_new, f_new, g_new, H_new, d, alpha
    """
    # Compute Newton direction by solving Hd = -g
    try:
        d = np.linalg.solve(H, -g)
    except np.linalg.LinAlgError:
        # If Hessian is singular, fall back to gradient descent
        d = -g

    # Solve TR subproblem
    d = ConjugateGradientSubproblem(f, g, H, problem, method)

    # Compute actual vs prediction reduction ratio
    rho = (f - problem.compute_f(x + d)) / (f - (f + g @ d + 0.5 * d.T @ H @ d))

    # Check if the step is acceptable
    if rho > method.options["c_1_tr"]:
        # Accept the step
        x_new = x + d
        f_new = problem.compute_f(x_new)
        g_new = problem.compute_g(x_new)

        # TODO: Update Hessian with SR1 formula
        H_new = problem.compute_H(x_new)

        # Update trust region radius
        if rho > method.options["c_2_tr"]:
            Delta = 2 * Delta

        return x_new, f_new, g_new, H_new, d, Delta
    else:
        # Reject the step and reduce the trust region radius
        Delta = 0.5 * Delta

    return x, f, g, H, d, Delta
