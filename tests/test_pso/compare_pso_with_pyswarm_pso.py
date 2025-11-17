MAX_ITER = 10
SWARM_SIZE = 10
EXP_SIZE = 10

import numpy as np
from pyswarm import pso
from pso.pso import PSO
from pso.entities import PSOParams
from pso.constants import *
from tabulate import tabulate

# Define test functions and their known optima
functions = [
    {
        'name': 'Sphere',
        'func': lambda x: np.sum(x**2),
        'lb': [-5, -5],
        'ub': [5, 5],
        'optimal_x': [0, 0],
        'optimal_f': 0
    },
    {
        'name': 'Rosenbrock',
        'func': lambda x: 100*(x[1]-x[0]**2)**2 + (1-x[0])**2,
        'lb': [-2, -1],
        'ub': [2, 3],
        'optimal_x': [1, 1],
        'optimal_f': 0
    },
    {
        'name': 'Rastrigin',
        'func': lambda x: 10*len(x) + np.sum(x**2 - 10*np.cos(2*np.pi*x)),
        'lb': [-5.12, -5.12],
        'ub': [5.12, 5.12],
        'optimal_x': [0, 0],
        'optimal_f': 0
    },
    {
        'name': 'Ackley',
        'func': lambda x: -20*np.exp(-0.2*np.sqrt(0.5*(x[0]**2 + x[1]**2))) - np.exp(0.5*(np.cos(2*np.pi*x[0]) + np.cos(2*np.pi*x[1]))) + np.e + 20,
        'lb': [-5, -5],
        'ub': [5, 5],
        'optimal_x': [0, 0],
        'optimal_f': 0
    }
]


def pyswarm_pso():
    results = []
    for f in functions:
        run_results = []
        for _ in range(EXP_SIZE):
            _, fopt = pso(
                f['func'], 
                f['lb'], 
                f['ub'], 
                swarmsize=SWARM_SIZE, 
                maxiter=MAX_ITER, 
                debug=False
            )
            run_results.append(fopt)
        mean = np.mean(run_results)
        std = np.std(run_results)
        results.append((mean, std))
    return results

def custom_pso():
    results = []
    for f in functions:
        run_results = []
        for _ in range(EXP_SIZE):
            config = PSOParams(
                max_iter=MAX_ITER,
                swarm_size=SWARM_SIZE,
                w_inertia=0.5,
                c_personal=0.5,
                c_social=1.5,
                c_global=0.5,
                jump_size=1.0,
                informant_selection=InformantSelect.STATIC_RANDOM,
                informant_count=2,
                boundary_handling=BoundHandling.CLIP,
                dims=len(f['lb']),
                boundary_min=f['lb'],
                boundary_max=f['ub'],
                target_fitness=None
            )
            pso_solver = PSO(config)
            _, fopt = pso_solver.optimize(f['func'])
            run_results.append(fopt)
        mean = np.mean(run_results)
        std = np.std(run_results)
        results.append((mean, std))
    return results


if __name__ == "__main__":
    pyswarm_results = pyswarm_pso()
    custom_results = custom_pso()
    headers = [f["name"] for f in functions]
    def format_mean_std(results):
        return [f"{mean:.4e}  ±  {std:.2e}" for (mean, std) in results]
    # Determine winner for each function (lower mean wins)
    winner_row = ["winner"]
    for i in range(len(functions)):
        mean_pyswarm = pyswarm_results[i][0]
        mean_custom = custom_results[i][0]
        if mean_pyswarm < mean_custom:
            winner_row.append("pyswarm")
        elif mean_custom < mean_pyswarm:
            winner_row.append("custom")
        else:
            winner_row.append("tie")
    table = [
        ["pyswarm"] + format_mean_std(pyswarm_results),
        ["custom"] + format_mean_std(custom_results),
        winner_row
    ]
    print("\nPSO Optimization Results (Mean ± Std over 10 runs)")
    print(tabulate(
        table,
        headers=["Method"] + headers,
        tablefmt="github"
    ))
    print("\nKnown Optimum Values:")
    for f in functions:
        print(f"  {f['name']}: x = {f['optimal_x']}, f(x) = {f['optimal_f']}")


