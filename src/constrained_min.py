import os
import numpy as np
import math
import matplotlib.pyplot as plt

SAVE_DIR = './saves/'

def save_plot(title):
    SAVE_PATH = os.path.join(SAVE_DIR, title)

    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    plt.savefig(SAVE_PATH)

class InteriorPointOptimizer:
    def __init__(self, func, ineq_constraints, eq_constraints_mat, eq_constraints_rhs):
        self.func = func
        self.ineq_constraints = ineq_constraints
        self.eq_constraints_mat = eq_constraints_mat
        self.eq_constraints_rhs = eq_constraints_rhs

    def minimize(self, x0):
        t = 1
        x = x0
        num_of_constraints = len(self.ineq_constraints)
        self.history = []

        while (num_of_constraints / t) > 1e-8:
            self.history.append({"x": x.copy(), "fx": self.func(x.copy())[0]})

            for _ in range(10):
                f_x, g_x, h_x = self._apply_log_barrier_(x, t)
                direction = self._find_direction_(g_x, h_x)
                step_size = self._wolfe_condition_with_backtracking_(x, f_x, g_x, direction)

                x = x + direction * step_size
                l = np.sqrt(np.dot(direction, np.dot(h_x, direction.T)))

                if 0.5 * (l ** 2) < 1e-8:
                    break

            t *= 10

        return x, self.func(x.copy())[0]
    
    def plot_evaluations_graph(self, title):
        values = [entry['fx'] for entry in self.history]
        fig, ax = plt.subplots(1, 1)
        x = np.linspace(0, len(values)-1, len(values))
        ax.plot(x, values, marker='.')
        ax.set_title(title)
        ax.set_xlabel('Iterations')
        ax.set_ylabel('Objective values')
        save_plot(title)
        plt.show()

    def plot_3d_results(self, title):
        path = [entry['x'] for entry in self.history]
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        path = np.array(path)

        ax.plot_trisurf([1, 0, 0], [0, 1, 0], [0, 0, 1], alpha=0.5)
        ax.plot(path[:, 0], path[:, 1], path[:, 2], label='Path')
        ax.scatter(path[-1][0], path[-1][1], path[-1][2], s=50, c='gold', marker='o', label='Candidate')
        ax.set_title(title)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        plt.legend()
        ax.view_init(45, 45)
        save_plot(title)
        plt.show()

    def plot_2d_results(self, title):
        path = [entry['x'] for entry in self.history]
        fig, ax = plt.subplots(1, 1)
        path = np.array(path)

        x = np.linspace(-1, 3, 1000)
        y = np.linspace(-2, 2, 1000)
        contraints_ineq = {
            'y=0': (x, x*0),
            'y=1': (x, x*0 + 1),
            'x=2': (y*0 + 2, y),
            'y=-x+1': (x, -x + 1)
        }

        for f, (x, y) in contraints_ineq.items():
            ax.plot(x, y, label=f)

        ax.fill([0, 2, 2, 1], [1, 1, 0, 0], label='Feasible region')
        ax.plot(path[:, 0], path[:, 1], c='k', label='Path')
        ax.scatter(path[-1][0], path[-1][1], s=50, c='gold', marker='o', label='Candidate')
        ax.set_title(title)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.legend()
        save_plot(title)
        plt.show()

    def _find_direction_(self, g_x, h_x):
        if self.eq_constraints_mat is not None:
            A = np.block([
                [h_x, self.eq_constraints_mat.T],
                [self.eq_constraints_mat, 0],
            ])
            b = np.block([[-g_x, 0]])
            x = np.linalg.solve(A, b.T).T[0]
            return x[0:self.eq_constraints_mat.shape[1]]
        
        return np.linalg.solve(h_x, -g_x)
    
    def _log_barrier_(self, x):
        dim = x.shape[0]
        log_f_x = 0
        log_g_x = np.zeros((dim,))
        log_h_x = np.zeros((dim, dim))

        for constraint in self.ineq_constraints:
            f_x, g_x, h_x = constraint(x)
            log_f_x += math.log(-f_x)
            log_g_x += (1.0 / -f_x) * g_x

            gradient = g_x / f_x
            dim = gradient.shape[0]
            tile = np.tile(gradient.reshape(dim, -1), (1, dim)) * np.tile(gradient.reshape(dim, -1).T, (dim, 1))
            log_h_x += (h_x * f_x - tile) / f_x ** 2

        return -log_f_x, log_g_x, -log_h_x
    
    def _apply_log_barrier_(self, x, t):
        f_x, g_x, h_x = self.func(x)
        log_f_x, log_g_x, log_h_x = self._log_barrier_(x)

        resulted_f_x = t*f_x + log_f_x
        resulted_g_x = t*g_x + log_g_x
        resulted_h_x = t*h_x + log_h_x

        return resulted_f_x, resulted_g_x, resulted_h_x
    
    def _wolfe_condition_with_backtracking_(self, x, f_x, g_x, direction, alpha=0.01, beta=0.5, max_iterations=10):
        step_size = 1
        for _ in range(max_iterations):
            current_f_x, _, _ = self.func(x + step_size * direction)

            if current_f_x <= f_x + alpha * step_size * g_x.dot(direction):
                break

            step_size *= beta

        return step_size