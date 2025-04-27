import project.problems.project_problems as functions


class Problem:
    def __init__(self, name, x0):
        self.name = name
        self.x0 = x0
        self.n = len(x0)
        match name:
            case "quad_10_10":
                self.compute_f = functions.quad_10_10_func
                self.compute_g = functions.quad_10_10_grad
                self.compute_H = functions.quad_10_10_Hess
            case "quad_10_1000":
                self.compute_f = functions.quad_10_1000_func
                self.compute_g = functions.quad_10_1000_grad
                self.compute_H = functions.quad_10_1000_Hess
            case "quad_1000_10":
                self.compute_f = functions.quad_1000_10_func
                self.compute_g = functions.quad_1000_10_grad
                self.compute_H = functions.quad_1000_10_Hess
            case "quad_1000_1000":
                self.compute_f = functions.quad_1000_1000_func
                self.compute_g = functions.quad_1000_1000_grad
                self.compute_H = functions.quad_1000_1000_Hess
            case "quartic_1":
                self.compute_f = functions.quartic_1_func
                self.compute_g = functions.quartic_1_grad
                self.compute_H = functions.quartic_1_Hess
            case "quartic_2":
                self.compute_f = functions.quartic_2_func
                self.compute_g = functions.quartic_2_grad
                self.compute_H = functions.quartic_2_Hess
            case "rosenbrock_2":
                self.compute_f = functions.rosenbrock_2_func
                self.compute_g = functions.rosenbrock_2_grad
                self.compute_H = functions.rosenbrock_2_Hess
            case "rosenbrock_100":
                self.compute_f = functions.rosenbrock_100_func
                self.compute_g = functions.rosenbrock_100_grad
                self.compute_H = functions.rosenbrock_100_Hess
            case "datafit_2":
                self.compute_f = functions.datafit_2_func
                self.compute_g = functions.datafit_2_grad
                self.compute_H = functions.datafit_2_Hess
            case "exponential":
                self.compute_f = functions.exponential_func
                self.compute_g = functions.exponential_grad
                self.compute_H = functions.exponential_Hess
            case "genhumps_5":
                self.compute_f = functions.genhumps_5_func
                self.compute_g = functions.genhumps_5_grad
                self.compute_H = functions.genhumps_5_Hess

            case _:
                raise ValueError("problem not defined!!!")


class Method:
    def __init__(self, name, **options):
        self.name = name
        self.options = options


class Options:
    def __init__(self, term_tol=1e-6, max_iterations=1e2):
        self.term_tol = term_tol
        self.max_iterations = max_iterations
