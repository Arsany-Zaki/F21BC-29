# Out-of-the-box PSO implementation from pyswarm library
# Test the implemented PSO against this one for verification
# It IS NOT used in the final experiments, it is only for testing purposes 

from .entities import PSOParams
from pyswarm import pso as pyswarm_pso

class PySwarmPSO:
	def __init__(self, config: PSOParams):
		self.config = config

	def optimize_with_given_config(self, fitness_func):
		xopt, fopt = pyswarm_pso(
			fitness_func,
			lb=self.config.boundary_min,
			ub=self.config.boundary_max,
			swarmsize=self.config.swarm_size,
			maxiter=self.config.max_iter,
			
			debug=False
		)
		return xopt, fopt

	def optimize_with_default_config(self, fitness_func):
		xopt, fopt = pyswarm_pso(
			fitness_func,
			lb=self.config.boundary_min,
			ub=self.config.boundary_max,
			debug=False
		)
		return xopt, fopt