import time
import numpy as np
import pandas as pd
from tqdm import tqdm

from optSolver import optSolver
from project.problems.project_problems import *  # your functions
from framework import Problem, Method, Options


problems = Problem("rosenbrock_2", np.array([-1.2, 1.0]))

method = Method(
    "ModifiedNewton",
    step_type="Wolfe",
    alpha=1,
    tau=0.5,
    c_1_ls=1e-4,
    c_2_ls=0.9,
    beta=1e-6,
)

options = Options(term_tol=1e-6, max_iterations=1e3)

x, f = optSolver(problems, method, options)

print(f"Solution: {x}")
print(f"Function value: {f}")
