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
            c1 = method.options.get("c_1_ls", 1e-4)

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
            c1 = method.options.get("c_1_ls", 1e-4)
            c2 = method.options.get("c_2_ls", 0.9)

            # Subroutine parameters
            alpha_low = method.options.get("alpha_low", 0)
            alpha_high = method.options.get("alpha_high", 1000)
            alpha = method.options.get("alpha", 1)
            c = method.options.get("c", 0.5)
            tol = method.options.get("alpha_tol", 1e-8)

            while True:
                x_new = x + alpha * d
                # Sufficient decrease condition
                if problem.compute_f(x_new) <= f + c1 * alpha * g.T @ d:
                    # Curvature condition
                    if problem.compute_g(x_new).T @ d >= c2 * g.T @ d:
                        break
                    else:
                        alpha_low = alpha
                else:
                    alpha_high = alpha

                if abs(alpha_high - alpha_low) < tol:
                    return alpha

                alpha = c * alpha_low + (1 - c) * alpha_high

        case _:
            raise ValueError("step type is not defined")

    return alpha


def ConjugateGradientSubproblem(f, g, H, Delta, problem, method):
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
            # solve ||z + τ p|| = Δ
            p_norm_sq = p.T @ p
            z_p = z.T @ p
            rad = np.sqrt(z_p**2 - p_norm_sq * (z.T @ z - Delta**2))
            tau = (-z_p + rad) / p_norm_sq
            return z + tau * p

        alpha = (r.T @ r) / (p.T @ B @ p)

        z_new = z + alpha * p

        # If outside of region
        if np.sqrt(z_new.T @ z_new) >= Delta:
            # solve ||z + τ p|| = Δ
            p_norm_sq = p.T @ p
            z_p = z.T @ p
            rad = np.sqrt(z_p**2 - p_norm_sq * (z.T @ z - Delta**2))
            tau = (-z_p + rad) / p_norm_sq
            return z + tau * p

        r_new = r + alpha * B @ p

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

    return x_new, f_new, g_new


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

    return x_new, f_new, g_new, H_new


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

    return x_new, f_new, g_new, H_new


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

    return x_new, f_new, g_new, H_new, n_skipped


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

    return x_new, f_new, g_new, n_skipped


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

    return x_new, f_new, g_new, H_new


# Trust region methods


# TRNewtonCG
def TRNewtonStep(x, f, g, H, Delta, problem, method, options):
    """Function that: (1) computes the TR Newton step; (2) updates the iterate; and,
        (3) computes the function and gradient at the new iterate

    Inputs:
        x, f, g, H, problem, method, options
    Outputs:
        x_new, f_new, g_new, H_new, d, Delta
    """
    # Load parameters
    p = method.options
    c1 = p.get("c_1_tr", 1e-3)  # accept if ρ > c1   (book η)
    c2 = p.get("c_2_tr", 0.75)  # enlarge if ρ > c2  (book ¾)
    gamma_inc = p.get("gamma_inc", 2.0)
    gamma_dec = p.get("gamma_dec", 0.5)
    Delta_max = p.get("Delta_max", 100.0)

    # Solve TR subproblem
    s = ConjugateGradientSubproblem(f, g, H, Delta, problem, method)

    # Reductions
    f_trial = problem.compute_f(x + s)
    ared = f - f_trial
    pred = -(g @ s + 0.5 * s @ H @ s)

    # model not trustworthy, decrease radius
    if pred <= 0:
        Delta = max(gamma_dec * Delta, 1e-6)
        return x, f, g, H, Delta

    rho = ared / pred

    # Check if the step is acceptable
    if rho > c1:
        # Accept the step
        x_new = x + s
        f_new = problem.compute_f(x_new)
        g_new = problem.compute_g(x_new)
        H_new = problem.compute_H(x_new)

        # Update trust region radius
        if rho > c2:
            Delta = min(gamma_inc * Delta, Delta_max)

        return x_new, f_new, g_new, H_new, Delta
    else:
        # Reject the step and reduce the trust region radius
        Delta = max(gamma_dec * Delta, 1e-6)

    return x, f, g, H, Delta


# TRSR1CG
def TRSR1Step(x, f, g, B, Delta, n_skipped, problem, method, options):
    """
    SR1 trust-region step that uses the SAME (c_1_tr, c_2_tr)
    semantics as your TRNewtonStep.
    """

    # Load parameters
    p = method.options
    c1 = p.get("c_1_tr", 1e-3)  # accept if ρ > c1   (book η)
    c2 = p.get("c_2_tr", 0.75)  # enlarge if ρ > c2  (book ¾)
    gamma_inc = p.get("gamma_inc", 2.0)
    gamma_dec = p.get("gamma_dec", 0.5)
    Delta_max = p.get("Delta_max", 100.0)
    epsilon_sy = p.get("epsilon_sy", 1e-8)

    # Solve TR subproblem
    s = ConjugateGradientSubproblem(f, g, B, Delta, problem, method)

    # Reductions
    f_trial = problem.compute_f(x + s)
    ared = f - f_trial
    pred = -(g @ s + 0.5 * s @ B @ s)

    # model not trustworthy, decrease radius
    if pred <= 0:
        Delta = max(gamma_dec * Delta, 1e-6)
        return x, f, g, B, Delta, n_skipped

    rho = ared / pred

    # Check if the step is acceptable
    if rho > c1:
        x_new = x + s
        f_new = f_trial
        g_new = problem.compute_g(x_new)

        # SR1 update with curvature safeguard
        y = g_new - g
        yBs = y - B @ s
        denom = s @ yBs

        # Check if we can update B
        if abs(denom) >= epsilon_sy * np.linalg.norm(s) * np.linalg.norm(yBs):
            B_new = B + np.outer(yBs, yBs) / denom
        else:
            B_new = B
            n_skipped += 1

        # radius update
        if rho > c2:
            Delta = min(gamma_inc * Delta, Delta_max)

        return x_new, f_new, g_new, B_new, Delta, n_skipped

    # Reject step and reduce trust region radius
    else:
        Delta = max(gamma_dec * Delta, 1e-6)
        return x, f, g, B, Delta, n_skipped


# def TRSR1Step(x, f, g, H, Delta, n_skipped, problem, method, options):
#     """Function that: (1) computes the TR Newton step; (2) updates the iterate; and,
#         (3) computes the function and gradient at the new iterate

#     Inputs:
#         x, f, g, H, problem, method, options
#     Outputs:
#         x_new, f_new, g_new, H_new, Delta, n_skipped
#     """
#     # Solve TR subproblem
#     d = ConjugateGradientSubproblem(f, g, H, Delta, problem, method)

#     # Compute actual vs prediction reduction ratio
#     rho = (f - problem.compute_f(x + d)) / (-(g.T @ d + 0.5 * d.T @ H @ d))
#     print(f"reduction: {rho}")
#     print(f"delta: {Delta}")
#     # Check if the step is acceptable
#     if rho > method.options["c_1_tr"]:
#         # Accept the step
#         x_new = x + d
#         f_new = problem.compute_f(x_new)
#         g_new = problem.compute_g(x_new)
#         print(f"norm_g {np.linalg.norm(g_new)}")

#         y = g_new - g
#         # remember s = d
#         # Check if we can update  Hessian
#         yHd = y - H @ d
#         if d.T @ (yHd) > method.options["epsilon_sy"] * np.linalg.norm(
#             d
#         ) * np.linalg.norm(yHd):
#             # Update Hessian with SR1 formula
#             H_new = H + np.outer(yHd, yHd) / (yHd.T @ d)
#         else:
#             H_new = H
#             n_skipped += 1

#         # Update trust region radius
#         if rho > method.options["c_2_tr"]:
#             Delta = 2 * Delta

#         return x_new, f_new, g_new, H_new, Delta, n_skipped
#     else:
#         # Reject the step and reduce the trust region radius
#         Delta = 0.5 * Delta

#     return x, f, g, H, Delta, n_skipped
