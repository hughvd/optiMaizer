import numpy as np
from optSolver import optSolver
from framework import Method, Options


def getQuadtricPenaltySubproblem(x0, base_problem, nu):
    """Function that returns problem, method, options for the Quadratic Penalty subproblem
        phi(x, nu) = f(x) + nu/2 * ||c(x)||^2

    Inputs:
        problem (struct)
    Outputs:
        phi_subproblem (struct), phi_method (struct), phi_options (struct)
    """

    # helper closure functions
    def phi_f(x):
        c = base_problem.compute_const(x)
        return base_problem.compute_f(x) + 0.5 * nu * np.dot(c, c)

    def phi_g(x):
        g = base_problem.compute_g(x)
        J = base_problem.compute_const_g(x)  # shape (m,n)
        c = base_problem.compute_const(x)
        return g + nu * J.T @ c

    def phi_H(x):
        Hf = base_problem.compute_H(x)
        J = base_problem.compute_const_g(x)
        c = base_problem.compute_const(x)
        # Hφ = Hf + ν (JᵀJ + Σ c_i ∇²c_i)
        Hphi = Hf + nu * (J.T @ J)
        Hconst = base_problem.compute_const_H(x)  # shape (m,n,n)
        for i, ci in enumerate(c):
            Hphi += nu * ci * Hconst[i]
        return Hphi

    # tiny wrapper
    class PhiProblem:
        def __init__(self):
            self.name = "QuadPenaltySub"
            self.x0 = x0
            self.n = base_problem.n
            self.compute_f = phi_f
            self.compute_g = phi_g
            self.compute_H = phi_H

    # Create the subproblem
    phi_problem = PhiProblem()
    phi_method = Method(
        "BFGS",
        step_type="Backtracking",
        alpha=1,
        tau=0.5,
        c_1_ls=1e-4,
        epsilon_sy=1e-6,
    )
    phi_options = Options(term_tol=1.0 / nu, max_iterations=1e3)

    return phi_problem, phi_method, phi_options


def QuadPenaltyStep(x, nu, problem, method, options):
    """Function that computes the next iterate using the Quadratic Penalty method

    Inputs:
        x (current iterate), f (function value), g (gradient), problem, method, options (structs)
    Outputs:
        x_new (next iterate), f_new (function value), number of subproblem iterations
    """
    # Form the subproblem
    phi_subproblem, phi_method, phi_options = getQuadtricPenaltySubproblem(
        x, problem, nu
    )

    # Compute the next iterate by solving the subproblem
    x_new, f_new, history = optSolver(phi_subproblem, phi_method, phi_options)

    return x_new, f_new, history["iterations"][-1]
