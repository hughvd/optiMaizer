import numpy as np

def compute_step_size(X, y, w, d, loss, g, k, problem, method):
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
            w_new = w + alpha * d
            loss_new = problem.compute_f(X, y, w_new)
            
            # Backtracking
            while loss_new > loss + c1 * alpha * g @ d:
                alpha = tau * alpha
                w_new = w + alpha * d
                loss_new = problem.compute_f(X, y, w_new)
                
                # Break if step size becomes too small
                if alpha < 1e-10:
                    break
        case "Diminishing":
            alpha = method.options["alpha"] / k
        case _:
            raise ValueError("step type is not defined")
    
    return alpha

def GDStep(X, y, w, loss, g, k, problem, method, options):
    """Function that: (1) computes the loss; (2) computes the gradient; and,
        (3) updates the weights

    Inputs:
        X, y, w, f, g, k, problem, method, options
    Outputs:
        w_new
    """
    # Set the search direction d to be -g
    d = -g

    # Compute step size
    alpha = compute_step_size(X, y, w, d, loss, g, k, problem, method)

    # Update
    w_new = w + alpha * d

    return w_new