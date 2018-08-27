# import os
import time
import numpy            as np
import pandas           as pd
from tqdm               import tqdm
from multiprocessing    import Pool

from optimize       import optimize
from evolution      import (DifferentialEvolution, OppositionDifferentialEvolution,
                            DifferentialEvolutionSimple, OppositionDifferentialEvolutionSimple,
                            ParticleSwarmOptimizationSimple, GOParticleSwarmOptimizationSimple)
from vis_functions  import plot_evolution
import dirs

dim     = 2
popSize = 5
func_id = 1

# TEST EVOLUTION ALGS
# alg = GOParticleSwarmOptimizationSimple(func_id=1, dim=dim, pop_size=popSize)
# print(alg.population)
#
# alg.generation()
# alg.generation()
# print(alg.population)

## TEST PLOT FUNCTIONS
# tablePath = dirs.tables+"DE/"+"DE_table2_F14_dim10.xlsx"
# plot_evolution(tablePath, fig_name="auto", save=True)
