import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
import numpy as np
from src.constrained_min import InteriorPointOptimizer 
from tests.examples import *

def run_test(name, objective_methods, ineqs, eqs, x0):
    ipo = InteriorPointOptimizer(objective_methods, ineqs, eqs, 0)
    candidate, objective = ipo.minimize(x0)

    print('-' * 10, name, '-' * 10)
    print('Candidate:', candidate)
    print('Objective value:', objective)
    print('Inequality constraints for candidate:')
    for constraint in ineqs:
        print(constraint, ':', constraint(candidate))
    if eqs is not None:
        print('Equality constraints for candidate:', (quadratic_eq_constraints * candidate).sum())
    print('-' * 24)

    ipo.plot_evaluations_graph(f'Objective values per outer iteration number - {name}')

    if len(x0) == 3:
        ipo.plot_3d_results(f'Feasible region and path taken by the algorithm - {name}')
    else:
        ipo.plot_2d_results(f'Feasible region and path taken by the algorithm - {name}')


class TestMinimize(unittest.TestCase):
    def test_qp(self):
        run_test('QP', quadratic_objective, quadratic_ineq_constaints, quadratic_eq_constraints, np.array([0.1, 0.2, 0.7]))

    def test_lp(self):
        run_test('LP', ceiling_objective, ceiling_ineq_constraints, None, np.array([0.5, 0.75]))


if __name__ == '__main__':
    try:
        unittest.main()
    except SystemExit:
        pass
