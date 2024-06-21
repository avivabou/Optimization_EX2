import numpy as np

def quadratic_objective(x):
    f = x[0] ** 2  + x[1] ** 2 + (x[2] + 1) ** 2
    g = np.array([2 * x[0], 2 * x[1], 2 * x[2] + 2])
    h = np.array([
        [2, 0, 0],
        [0, 2, 0],
        [0, 0, 2]
    ])
    return f, g, h

def quadratic_ineq1(x):
    f = -x[0]
    g = np.array([-1, 0, 0])
    h = np.array([
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]
    ])
    return f, g, h

def quadratic_ineq2(x):
    f = -x[1]
    g = np.array([0, -1, 0])
    h = np.array([
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]
    ])
    return f, g, h

def quadratic_ineq3(x):
    f = -x[2]
    g = np.array([0, 0, -1])
    h = np.array([
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]
    ])
    return f, g, h

quadratic_ineq_constaints = [quadratic_ineq1, quadratic_ineq2, quadratic_ineq3]
quadratic_eq_constraints = np.array([1, 1, 1]).reshape(1, 3)

def ceiling_objective(x):
    f = -x[0] - x[1]
    g = np.array([-1, -1])
    h = np.array([
        [0,0],
        [0,0]
    ])
    return f, g, h

def ceiling_ineq1(x):
    # y <=1
    f = x[1] -1
    g = np.array([0, 1])
    h = np.array([
        [0, 0], 
        [0, 0]
    ])
    return f, g, h

def ceiling_ineq2(x):
    # x <=2 
    f = x[0] -2
    g = np.array([1, 0])
    h = np.array([
        [0, 0], 
        [0, 0]
    ])
    return f, g, h

def ceiling_ineq3(x):
    # y >=0
    f = -x[1]
    g = np.array([0, -1])
    h = np.array([
        [0, 0],
        [0, 0]
    ])
    return f, g, h
 
def ceiling_ineq4(x):
    # 𝑦 ≥ −𝑥 + 1 = -x -y +1 <=0
    f = -x[0] - x[1] + 1
    g = np.array([-1, -1])
    h = np.array([
        [0, 0],
        [0, 0]
    ])
    return f, g, h

ceiling_ineq_constraints = [ceiling_ineq1, ceiling_ineq2, ceiling_ineq3, ceiling_ineq4]