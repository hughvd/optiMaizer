import time
import numpy as np
import pandas as pd
from tqdm import tqdm

from optSolver import optSolver
from project.problems.project_problems import *  # your functions
from framework import Problem, Method, Options  # wherever you defined these


def count_calls(fn, counter_dict, key):
    """Returns a wrapped fn that bumps counter_dict[key] on every call."""

    def wrapped(x, *args, **kwargs):
        counter_dict[key] += 1
        return fn(x, *args, **kwargs)

    return wrapped


# 1. Build problem list
np.random.seed(0)
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
problems = [Problem(name, x0) for name, x0 in problem_specs]

# Method list
methods = [
    Method("GradientDescent", step_type="Backtracking", alpha=1, tau=0.5, c_1_ls=1e-4),
    Method(
        "GradientDescent", step_type="Wolfe", alpha=1, tau=0.5, c_1_ls=1e-4, c_2_ls=0.9
    ),
    Method(
        "ModifiedNewton",
        step_type="Backtracking",
        alpha=1,
        tau=0.5,
        c_1_ls=1e-4,
        beta=1e-6,
    ),
    Method(
        "ModifiedNewton",
        step_type="Wolfe",
        alpha=1,
        tau=0.5,
        c_1_ls=1e-4,
        c_2_ls=0.9,
        beta=1e-6,
    ),
    Method("TRNewtonCG", c_1_tr=1e-3, c_2_tr=0.75, term_tol_CG=1e-10),
    Method("TRSR1CG", c_1_tr=1e-3, c_2_tr=0.75, epsilon_sy=1e-8, term_tol_CG=1e-10),
    Method(
        "BFGS", step_type="Backtracking", alpha=1, tau=0.5, c_1_ls=1e-4, epsilon_sy=1e-6
    ),
    Method(
        "BFGS",
        step_type="Wolfe",
        alpha=1,
        tau=0.5,
        c_1_ls=1e-4,
        c_2_ls=0.9,
        epsilon_sy=1e-6,
    ),
    Method(
        "DFP", step_type="Backtracking", alpha=1, tau=0.5, c_1_ls=1e-4, epsilon_sy=1e-6
    ),
    Method(
        "DFP",
        step_type="Wolfe",
        alpha=1,
        tau=0.5,
        c_1_ls=1e-4,
        c_2_ls=0.9,
        epsilon_sy=1e-6,
    ),
]

method_names = [
    "GradientDescent,",
    "GradientDescentW",
    "Newton",
    "NewtonW",
    "TRNewtonCG",
    "TRSR1CG",
    "BFGS",
    "BFGSW",
    "DFP",
    "DFPW",
]

# ————————————————
# 3. Loop and collect
records = []

for prob in tqdm(problems):
    # create fresh counters for each problem
    counters = {"f_evals": 0, "g_evals": 0}

    # wrap the compute functions
    prob.compute_f = count_calls(prob.compute_f, counters, "f_evals")
    prob.compute_g = count_calls(prob.compute_g, counters, "g_evals")

    for meth in tqdm(methods):
        opts = Options(term_tol=1e-6, max_iterations=1e3)

        t0 = time.time()
        try:
            x_star, f_star, history = optSolver(prob, meth, opts)
            cpu = time.time() - t0

            iters = history["iterations"][-1]
            fcount = counters["f_evals"]
            gcount = counters["g_evals"]
            status = "ok"

        except Exception as e:
            cpu, iters, fcount, gcount = None, None, None, None
            status = f"fail: {e!r}"

        records.append(
            {
                "problem": prob.name,
                "method": meth.name,
                "iterations": iters,
                "f_evals": fcount,
                "grad_evals": gcount,
                "cpu_seconds": cpu,
                "status": status,
            }
        )

# 4. Build DataFrame & preview
df = pd.DataFrame(records)
print(df)
df.to_csv("results.csv", index=False)

# 5. (When you’re ready) pivot for your table:
#   df_iters = df.pivot(index="problem", columns="method", values="iterations")
#   df_f    = df.pivot(index="problem", columns="method", values="f_evals")
#   df_g    = df.pivot(index="problem", columns="method", values="grad_evals")
#   df_cpu  = df.pivot(index="problem", columns="method", values="cpu_seconds")
