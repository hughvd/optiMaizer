# OptiMaizer

This package provides a framework for solving optimization problems using various algorithms. This code was written for the class Math 562 / IOE 511: Continuous Optimization. 

---

## Features

### Core Components

1. **`optSolver`**:
   - The main function that runs a chosen optimization algorithm on a specified problem.
   - Inputs:
     - `problem`: Defines the optimization problem (objective function, gradient, Hessian, etc.).
     - `method`: Specifies the optimization algorithm and its parameters.
     - `options`: Contains termination criteria and other settings.
   - Outputs:
     - Final iterate (`x`), final function value (`f`), and additional history (if applicable).

2. **`algorithms.py`**:
   - Contains implementations of various optimization algorithms, such as:
     - Gradient Descent
     - Newton's Method
     - Modified Newton
     - BFGS
     - L-BFGS
   - Supports different step size strategies, including backtracking line search.

3. **`functions.py`**:
   - Provides test functions, their gradients, and Hessians, including:
     - Rosenbrock function
     ```math
     f(x) = (1 - x_0)^2 + 100 (x_1 - x_0^2)^2
     ```
     - Quadratic function
     ```math
     f(x) = \frac{1}{2} x^T A x - b^T x + c
     ```
     - Function 2
     ```math
     f(x) = \sum_{i=1}^{3} \left( y_i - x_0 \cdot \left( 1 - x_1^{i+1} \right) \right)^2
     ```
     - Function 3
     ```math
     f(x) = \frac{\exp(x_0) - 1}{\exp(x_0) + 1} + 0.1 \exp(-x_0) + \sum_{i=1}^{n} (x_i - 1)^4
     ```


---

## Installation

Clone the repository and install the required dependencies:

```bash
git clone <repository-url>
cd framework_PYTHON
pip install -r requirements.txt
```

---

## Usage

### Example Workflow

1. **Define the Problem**:
   Use the `Problem` class to specify the optimization problem, including the function, gradient, and Hessian.

2. **Choose an Algorithm**:
   Use the `Method` class to select an optimization algorithm and configure its parameters.

3. **Set Options**:
   Use the `Options` class to define termination criteria and other settings.

4. **Run the Solver**:
   Call `optSolver` with the problem, method, and options to solve the optimization problem.

### Example Code

Below is an example of how to use the package, as demonstrated in script.ipynb:

```python
import numpy as np
from optSolver import optSolver
from functions import rosen_func, rosen_grad, rosen_Hess

# Define the problem
class Problem:
    def __init__(self, name, x0):
        self.name = name
        self.x0 = x0
        if name == "Rosenbrock":
            self.compute_f = rosen_func
            self.compute_g = rosen_grad
            self.compute_H = rosen_Hess

problem = Problem(name="Rosenbrock", x0=np.array([1.2, 1.2]))

# Define the method
class Method:
    def __init__(self, name, step_type, alpha, tau, c1):
        self.name = name
        self.step_type = step_type
        self.alpha = alpha
        self.tau = tau
        self.c1 = c1

method = Method(name="GradientDescent", step_type="Backtracking", alpha=1, tau=0.5, c1=1e-4)

# Define options
class Options:
    def __init__(self, term_tol=1e-6, max_iterations=100):
        self.term_tol = term_tol
        self.max_iterations = max_iterations

options = Options(term_tol=1e-6, max_iterations=100)

# Solve the problem
x, f, _ = optSolver(problem, method, options)

print("Optimal solution:", x)
print("Optimal function value:", f)
```

---

## File Structure

- **`functions.py`**: Contains test functions, gradients, and Hessians.
- **`algorithms.py`**: Implements optimization algorithms.
- **`optSolver.py`**: Main solver function that integrates problems, methods, and options.
- **`script.ipynb`**: Example notebook demonstrating how to use the package.
- **`requirements.txt`**: List of dependencies.

---

## Dependencies

- Python 3.x
- NumPy
- SciPy (for loading `.mat` files in examples)

Install dependencies using:

```bash
pip install numpy scipy
```