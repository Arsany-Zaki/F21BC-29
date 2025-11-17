from dataclasses import dataclass
from typing import List
from pso.constants import *

@dataclass
class PSOParams:
    max_iter: int

    swarm_size: int
    w_inertia: float    
    c_personal: float   
    c_social: float     # informant influence
    c_global: float     # global best influence
    jump_size: float
    informant_selection: InformantSelect
    informant_count: int
    boundary_handling: BoundHandling

    dims: int
    boundary_min: List[float]
    boundary_max: List[float]
    target_fitness: float | None


