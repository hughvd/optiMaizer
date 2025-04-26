import numpy as np

# ─────────────────────────── Problem 1 ───────────────────────────
#   min x1 + x2
#   s.t. x1^2 + x2^2 - 2 = 0


def p1_func(x):
    # x: array of length 2
    return x[0] + x[1]


def p1_grad(x):
    return np.array([1.0, 1.0])


def p1_Hess(x):
    # constant zero Hessian
    return np.zeros((2, 2))


def p1_c(x):
    # single constraint c(x) = x1^2 + x2^2 - 2
    return np.array([x[0] ** 2 + x[1] ** 2 - 2.0])


def p1_jac_c(x):
    # shape (m=1, n=2)
    return np.array([[2.0 * x[0], 2.0 * x[1]]])


def p1_Hess_c(x):
    # list of m=1 Hessian matrices, each shape (2,2)
    H = np.array([[2.0, 0.0], [0.0, 2.0]])
    return [H]


# ─────────────────────────── Problem 2 ───────────────────────────
#   min exp(x1 x2 x3 x4 x5) - ½(x1^3 + x2^3 + 1)^2
#   s.t.
#     c1(x) = x1^2 + x2^2 + x3^2 + x4^2 + x5^2 - 10 = 0
#     c2(x) = x2 x3 - 5 x4 x5 = 0
#     c3(x) = x1^3 + x2^3 + 1 = 0


def p2_func(x):
    x1, x2, x3, x4, x5 = x
    p = x1 * x2 * x3 * x4 * x5
    g = x1**3 + x2**3 + 1.0
    return np.exp(p) - 0.5 * g**2


def p2_grad(x):
    x1, x2, x3, x4, x5 = x
    # helper quantities
    p = x1 * x2 * x3 * x4 * x5
    E = np.exp(p)
    # ∇(e^p) = e^p * (∂p/∂xi)
    dp = np.array([p / x1, p / x2, p / x3, p / x4, p / x5])
    # g = x1^3 + x2^3 + 1
    g = x1**3 + x2**3 + 1.0
    dg = np.array([3 * x1**2, 3 * x2**2, 0.0, 0.0, 0.0])
    # total gradient
    return E * dp - g * dg


def p2_Hess(x):
    x1, x2, x3, x4, x5 = x
    p = x1 * x2 * x3 * x4 * x5
    E = np.exp(p)
    # first‐ and second‐partials of p
    dp = np.array([p / x1, p / x2, p / x3, p / x4, p / x5])
    # second partial ∂²p/∂xi∂xj = p/(xi xj) for i≠j, 0 for i=j
    # build tensor of those
    H_p = np.zeros((5, 5))
    for i in range(5):
        for j in range(5):
            if i != j:
                H_p[i, j] = p / (x[i] * x[j])
    # Hessian of e^p: E*(dp_i dp_j + H_p[i,j])
    H_e = E * (np.outer(dp, dp) + H_p)

    # now the “−½ g²” piece
    g = x1**3 + x2**3 + 1.0
    dg = np.array([3 * x1**2, 3 * x2**2, 0.0, 0.0, 0.0])
    # Hessian of g: diag([6*x1, 6*x2, 0,0,0])
    H_g = np.diag([6 * x1, 6 * x2, 0.0, 0.0, 0.0])
    # Hessian of −½ g² = −(dg dgᵀ + g * H_g)
    H_q = -(np.outer(dg, dg) + g * H_g)

    return H_e + H_q


def p2_c(x):
    x1, x2, x3, x4, x5 = x
    return np.array(
        [
            x1**2 + x2**2 + x3**2 + x4**2 + x5**2 - 10.0,
            x2 * x3 - 5.0 * x4 * x5,
            x1**3 + x2**3 + 1.0,
        ]
    )


def p2_jac_c(x):
    # shape (m=3, n=5)
    x1, x2, x3, x4, x5 = x
    # each row is ∇cᵢ
    return np.array(
        [
            [2 * x1, 2 * x2, 2 * x3, 2 * x4, 2 * x5],  # ∇c1
            [0.0, x3, x2, -5 * x5, -5 * x4],  # ∇c2
            [3 * x1**2, 3 * x2**2, 0.0, 0.0, 0.0],  # ∇c3
        ]
    )


def p2_Hess_c(x):
    x1, x2, x3, x4, x5 = x
    # Hessian of c1: 2*I₅
    H1 = 2.0 * np.eye(5)
    # Hessian of c2: only ∂²/∂x2∂x3 = ∂²/∂x3∂x2 = 1, ∂²/∂x4∂x5 = ∂²/∂x5∂x4 = -5
    H2 = np.zeros((5, 5))
    H2[1, 2] = H2[2, 1] = 1.0
    H2[3, 4] = H2[4, 3] = -5.0
    # Hessian of c3: diag([6*x1, 6*x2, 0,0,0])
    H3 = np.diag([6 * x1, 6 * x2, 0.0, 0.0, 0.0])

    return [H1, H2, H3]
