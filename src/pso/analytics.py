# Printing module for PSO runs and iterations
# It is mainly used for countling iterations and printing global best fitness values

from typing import List
import config.global_config as gc

class Analytics_Swarm_Iteration:
    def __init__(self, global_best_fitness: float):
        self.global_best_fitness = global_best_fitness
    
class Analytics_PSO_Run:
    def __init__(self):
        self.analytics_swarm_runs: List[Analytics_Swarm_Iteration] = []
        if(self.is_debug_mode()):
            print('PSO Run Initialized')
            
    def add_global_best_fitness(self, fitness: float):
        new = (len(self.analytics_swarm_runs) == 0 or 
               self.analytics_swarm_runs[-1].global_best_fitness != fitness)
        self.analytics_swarm_runs.append(Analytics_Swarm_Iteration(global_best_fitness=fitness))
        if(self.is_debug_mode()):
            new_text = " (new)" if new else ""
            print(f'Iteration {len(self.analytics_swarm_runs):4d}:   Global Best Fitness = {fitness:12.6f}   {new_text}')

    def is_debug_mode(self) -> bool:
        return gc.GC_PSO_PRINT