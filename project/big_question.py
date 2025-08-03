import time
import numpy as np
import pandas as pd
from tqdm import tqdm

from optSolver import optSolver
from framework import Problem, Method, Options


def count_calls(fn, counter_dict, key):
    """Returns a wrapped fn that bumps counter_dict[key] on every call."""
    # Keep track of original function if already wrapped
    original_fn = getattr(fn, "_original_fn", fn)

    def wrapped(x, *args, **kwargs):
        counter_dict[key] += 1
        return original_fn(x, *args, **kwargs)

    # Store the original function to prevent multiple wrappings if called repeatedly
    wrapped._original_fn = original_fn
    return wrapped


# Problems
np.random.seed(0)  # For reproducibility of base points
deg70 = np.deg2rad(70)
quartic_x0 = np.array([np.cos(deg70), np.sin(deg70), np.cos(deg70), np.sin(deg70)])

problem_specs = [
    ("quad_10_10", np.random.rand(10) * 20 - 10),
    ("quad_10_1000", np.random.rand(10) * 20 - 10),
    ("quad_1000_10", np.random.rand(1000) * 20 - 10),
    ("quad_1000_1000", np.random.rand(1000) * 20 - 10),
    ("quartic_1", quartic_x0),
    ("quartic_2", quartic_x0),
    ("rosenbrock_2", np.array([-1.2, 1.0])),
    ("rosenbrock_100", np.concatenate(([-1.2, 1.0], np.ones(98)))),
    ("datafit_2", np.array([1.0, 1.0])),
    # for exponential, same compute_f handles any dim:
    ("exponential", np.concatenate(([1.0], np.zeros(9)))),  # 10-dim
    ("exponential", np.concatenate(([1.0], np.zeros(99)))),  # 100-dim
    ("genhumps_5", np.full(5, 506.2) * np.array([-1, 1, -1, 1, -1])),
]

# Methods
memory_sizes = [1, 10, 20, 40, 60, 80, 100]
methods_to_test = []
method_base_params = {
    "step_type": "Backtracking",
    "alpha": 1,
    "tau": 0.5,
    "c_1_ls": 1e-4,
    "epsilon_sy": 1e-6,
}

for m_size in memory_sizes:
    method_name = f"L-BFGS"
    methods_to_test.append(
        Method(
            "L-BFGS",
            step_type="Backtracking",
            alpha=1,
            tau=0.5,
            c_1_ls=1e-4,
            epsilon_sy=1e-6,
            memory_size=m_size,
        )
    )


options = Options(term_tol=1e-6, max_iterations=1000)
results_list = []
np.random.seed(1)

print("Starting L-BFGS Memory Experiment...")
for prob_name, x0 in tqdm(problem_specs, desc="Problems"):
    dim = len(x0)
    problem_obj = Problem(prob_name, x0)

    original_compute_f = problem_obj.compute_f
    original_compute_g = problem_obj.compute_g

    # Use a fresh problem instance
    current_problem = Problem(prob_name, x0)

    current_problem.compute_f = original_compute_f
    current_problem.compute_g = original_compute_g

    for method in tqdm(methods_to_test):
        # Reset counters for each run (problem, start_point, method)
        counters = {"f_evals": 0, "g_evals": 0}

        # Wrap compute functions for the current problem instance
        current_problem.compute_f = count_calls(
            current_problem.compute_f, counters, "f_evals"
        )
        current_problem.compute_g = count_calls(
            current_problem.compute_g, counters, "g_evals"
        )

        try:
            t0 = time.time()
            x_star, f_star, history = optSolver(current_problem, method, options)
            cpu = time.time() - t0

            iters = history.get("iterations", [np.nan])[-1]
            f_evals = counters["f_evals"]
            g_evals = counters["g_evals"]
            converged = history.get("converged", False)
            n_skipped = history.get("n_skipped", 0)

            norm_g_final = history.get("norm_g", [np.nan])[-1]
            record = {
                "problem": prob_name,
                "dimension": dim,
                "method": "L-BFGS",
                "memory_size": method.options["memory_size"],
                "iterations": iters,
                "f_evals": f_evals,
                "g_evals": g_evals,
                "f_final": f_star,
                "norm_g_final": norm_g_final,
                "cpu_seconds": cpu,
                "converged": converged,
                "n_skipped": n_skipped,
            }
            results_list.append(record)

        except Exception as e:
            print(
                f"\nError during run: Problem={prob_name}, Memory={method.options['memory_size']}"
            )
            print(f"Exception: {e}")
            # Record failure
            record = {
                "problem": prob_name,
                "dimension": dim,
                "method": "L-BFGS",
                "memory_size": method.options["memory_size"],
                "iterations": np.nan,
                "f_evals": counters["f_evals"],
                "g_evals": counters["g_evals"],
                "f_final": np.nan,
                "norm_g_final": np.nan,
                "cpu_seconds": time.time() - t0,
                "converged": False,
                "n_skipped": np.nan,
                "error": str(e),
            }
            results_list.append(record)

# Save Results
print("\nExperiment finished. Compiling results...")
df_results = pd.DataFrame(results_list)

print("Results preview:")
print(df_results.head())

output_filename = "lbfgs_memory_experiment_results.csv"
try:
    df_results.to_csv(output_filename, index=False)
    print(f"Results saved to {output_filename}")
except Exception as e:
    print(f"Error saving results to CSV: {e}")
