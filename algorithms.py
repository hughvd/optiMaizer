""" IOE 511/MATH 562, University of Michigan
Code template provided by: Albert S. Berahas & Jiahao Shi
Implemented by: Hugh Van Deventer
"""
import numpy as np
import scipy

def compute_step_size(x, d, f, g, problem, method):
    """Computes step size based on specified method"""
    match method.options["step_type"]:
        case "Constant":
            alpha = method.options["constant_step_size"]
        case "Backtracking":
            # Get params
            alpha = method.options["alpha"]
            tau = method.options["tau"]
            c1 = method.options["c1"]
            
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
        case _:
            raise ValueError("step type is not defined")
    
    return alpha

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
    beta = method.options['beta']

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
            eta = max(2*eta, beta)

    # Check if we successfully computed the cholesky factorization
    if not success:
        raise RuntimeError(f"Failed to compute Cholesky factorization after {max_iterations} iterations")

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
    if s @ y > method.options['epsilon_sy']*np.linalg.norm(s)*np.linalg.norm(y):
        rho = 1 / (s @ y)
        I = np.eye(len(x))
        H_new = (I - rho * np.outer(s, y)) @ H @ (I - rho * np.outer(y, s)) + rho * np.outer(s, s)
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
    for i in range(len(s_list)-1, -1, -1):
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

    if s @ y >= method.options['epsilon_sy']*np.linalg.norm(s)*np.linalg.norm(y):
        s_list.append(s)
        y_list.append(y)
    else:
        n_skipped += 1

    return x_new, f_new, g_new, d, alpha, n_skipped