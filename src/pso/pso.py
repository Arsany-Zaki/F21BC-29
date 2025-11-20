# PSO algorithm using informant-based social structure
# based on BC course week7 lecture

from .entities import PSOParams
import numpy as np
from typing import Callable, Tuple
from pso.constants import *
from pso.analytics import Analytics_PSO_Run

class PSO:
    def __init__(self, config: PSOParams):
        self.analytics = Analytics_PSO_Run()
        self.config = config
        self.rng = np.random.RandomState()

        self.dims = config.dims
        # Use boundary_min and boundary_max from config
        self.boundary_min = np.array(config.boundary_min)
        self.boundary_max = np.array(config.boundary_max)
        self.boundary_range = self.boundary_max - self.boundary_min
        self.vel_max = np.abs(self.boundary_max - self.boundary_min)

        self._initialize_swarm()

    # Lecture Algorithm Line 1 to 10 
    def _initialize_swarm(self):
        
        self.positions = self.rng.uniform(self.boundary_min, self.boundary_max, size=(self.config.swarm_size, self.dims))
        self.fitness = np.full(self.config.swarm_size, np.nan)

        vhigh = np.abs(self.boundary_max - self.boundary_min)
        vlow = -vhigh
        self.velocities = self.rng.uniform(vlow, vhigh, size=(self.config.swarm_size, self.dims))

        self.pbest_pos = self.positions.copy()
        self.pbest_fit = np.full(self.config.swarm_size, np.inf)
        self.sbest_pos = np.full((self.config.swarm_size, self.dims), np.nan)
        self.sbest_fit = np.full(self.config.swarm_size, np.inf)
        self.gbest_pos = None
        self.gbest_fit = np.inf

        self.informants = np.full((self.config.swarm_size, self.config.informant_count), np.nan, dtype=float)
    
    def _assess_fitness(self, fitness_func: Callable) -> None:
        for i, pos in enumerate(self.positions):
            self.fitness[i] = fitness_func(pos)
    
    def _update_personal_bests(self):
        improved = self.fitness < self.pbest_fit
        self.pbest_pos[improved] = self.positions[improved].copy()
        self.pbest_fit[improved] = self.fitness[improved]
    
    def _update_global_best(self):
        best_idx = np.argmin(self.pbest_fit)
        best_fit = self.pbest_fit[best_idx]
        
        if self.gbest_pos is None or best_fit < self.gbest_fit:
            self.gbest_pos = self.pbest_pos[best_idx].copy()
            self.gbest_fit = best_fit
    
    def _update_social_best(self):
        self._update_informants()
        
        for i in range(self.config.swarm_size):
            informant_ids = self.informants[i].astype(int)  # Convert to int for indexing
            
            # Find best informant
            best_idx = informant_ids[0]
            best_fit = self.pbest_fit[best_idx]
            
            for idx in informant_ids:
                if self.pbest_fit[idx] < best_fit:
                    best_fit = self.pbest_fit[idx]
                    best_idx = idx
            
            self.sbest_pos[i] = self.pbest_pos[best_idx]
            self.sbest_fit[i] = self.pbest_fit[best_idx]
    
    def _update_informants(self):
        if self.config.informant_count >= self.config.swarm_size:
            raise ValueError(f"informant_count ({self.config.informant_count}) must be less than swarm_size ({self.config.swarm_size})")
        
        n_particles = self.config.swarm_size
        
        if self.config.informant_selection == InformantSelect.STATIC_RANDOM:
            # Only set informants if they haven't been set yet (check if all elements are NaN)
            if np.isnan(self.informants).all():  # If all elements are NaN, none are initialized
                for i in range(n_particles):
                    candidates = [j for j in range(n_particles) if j != i]
                    selected = self.rng.choice(candidates, size=self.config.informant_count, replace=False)
                    self.informants[i] = selected
            # If already set, don't change them (static behavior)
            
        elif self.config.informant_selection == InformantSelect.DYNAMIC_RANDOM:
            # Always update informants for dynamic behavior
            for i in range(n_particles):
                candidates = [j for j in range(n_particles) if j != i]
                selected = self.rng.choice(candidates, size=self.config.informant_count, replace=False)
                self.informants[i] = selected
                
        elif self.config.informant_selection == InformantSelect.SPATIAL_PROXIMITY:
            # Always update informants based on current positions
            for i in range(n_particles):
                # Calculate distances to all other particles
                distances = []
                for j in range(n_particles):
                    if j != i:
                        dist = np.linalg.norm(self.positions[i] - self.positions[j])
                        distances.append((dist, j))
                
                # Sort by distance and select closest neighbors
                distances.sort(key=lambda x: x[0])
                selected = np.array([distances[k][1] for k in range(self.config.informant_count)])
                self.informants[i] = selected
        
        else:
            supported_options = [e.value for e in InformantSelect]
            raise ValueError(
                f"Unknown informant selection: {self.config.informant_selection}. "
                f"Supported: {supported_options}"
            )

    def _update_velocities(self):
        # Generate random coefficients for all particles and dimensions at once
        r1 = self.rng.uniform(0.0, 1.0, size=(self.config.swarm_size, self.dims))
        r2 = self.rng.uniform(0.0, 1.0, size=(self.config.swarm_size, self.dims))
        r3 = self.rng.uniform(0.0, 1.0, size=(self.config.swarm_size, self.dims))

        # Vectorized velocity update with correct PSO formula
        self.velocities = (
            self.config.w_inertia * self.velocities +
            r1 * self.config.c_personal * (self.pbest_pos - self.positions) +
            r2 * self.config.c_social * (self.sbest_pos - self.positions) +
            r3 * self.config.c_global * (self.gbest_pos - self.positions)
        )
    
    def _update_positions(self):
        self.positions += self.config.jump_size * self.velocities   
        self._apply_boundary_strategy()
    
    def _apply_boundary_strategy(self):
        if self.config.boundary_handling == BoundHandling.CLIP:
            # Vectorized clipping
            self.positions = np.clip(self.positions, self.boundary_min, self.boundary_max)
            
        elif self.config.boundary_handling == BoundHandling.REFLECT:
            # Handle reflection for each particle individually
            for i in range(self.config.swarm_size):
                for j in range(self.dims):
                    pos = self.positions[i, j]
                    min_bound = self.boundary_min[j]
                    max_bound = self.boundary_max[j]
                    
                    if pos < min_bound:
                        # Reflect below minimum
                        self.positions[i, j] = min_bound + (min_bound - pos)
                        self.velocities[i, j] *= -1
                    elif pos > max_bound:
                        # Reflect above maximum
                        self.positions[i, j] = max_bound - (pos - max_bound)
                        self.velocities[i, j] *= -1
            
            # Ensure reflected positions are still within bounds
            self.positions = np.clip(self.positions, self.boundary_min, self.boundary_max)
    
    def _check_termination(self, iter_count: int) -> bool:
        "Check termination conditions."
        if iter_count >= self.config.max_iter:
            return True
        if (self.config.target_fitness is not None and 
            self.gbest_fit is not None and 
            self.gbest_fit <= self.config.target_fitness):
            return True
        return False
    
    # main algorithm of PSO optimization
    def optimize(self, fitness_func: Callable[[np.ndarray], float]) -> Tuple[np.ndarray, float]:
        """Run PSO optimization."""
        iter_count = 0
        while not self._check_termination(iter_count):  # Lecture Algorithm Line 11 and Line 27 
            
            self._assess_fitness(fitness_func)  # Lecture Algorithm Line 13 
            self._update_personal_bests()          
            self._update_social_best()          
            self._update_global_best()          # Lecture Algorithm Line 12 - 15
            self._update_velocities()           # Lecture Algorithm Line 20 - 24
            self._update_positions()            # Lecture Algorithm Line 25 - 26
            
            iter_count += 1
            self.analytics.add_global_best_fitness(self.gbest_fit)
        return self.gbest_pos.copy(), self.gbest_fit
    
    def get_swarm_state(self) -> dict:
        """Get current swarm state for analysis."""
        return {
            'positions': self.positions.copy(),
            'velocities': self.velocities.copy(),
            'fitness': self.fitness.copy(),
            'pbest_pos': self.pbest_pos.copy(),
            'pbest_fit': self.pbest_fit.copy(),
            'sbest_pos': self.sbest_pos.copy(),
            'sbest_fit': self.sbest_fit.copy(),
            'gbest_pos': self.gbest_pos.copy() if self.gbest_pos is not None else None,
            'gbest_fit': self.gbest_fit
        }
