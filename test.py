import time
import numpy            as np
import pandas           as pd
from tqdm               import tqdm
from multiprocessing    import Pool

from optimize       import optimize
from evolution      import DifferentialEvolution, OppositionDifferentialEvolution
from vis_functions  import plot_evolution
import dirs

dim     = 2
popSize = 30
func_id = 1

# alg = DifferentialEvolution(dim=dim, pop_size=popSize)
alg = OppositionDifferentialEvolution(func_id=1, dim=dim, pop_size=popSize)
# print(alg.population.shape)
print(alg.population)
# alg.mutation(alg.mutation_rand_1)
# # print(alg.mutatedPopulation)
# alg.crossover_binomial(alg.mutatedPopulation)
# # print(alg.trialPopulation)
# alg.generation_jumping()


# index = 2
# targetVector = alg.population.iloc[index, :]
# for i in range(10):
#
alg.generation()
# print(alg.population)
alg.generation()
# # # alg.crossover_binomial(alg.mutation())
print(alg.population)

# tablePath = dirs.tables+"DE/"+"DE_table2_F14_dim10.xlsx"
# plot_evolution(tablePath, fig_name="auto", save=True)

# import pygmo    as pg
#
# prob = pg.problem(pg.schwefel(30))
#
# alg = pg.algorithm(pg.sade(gen=100))
#
# arch = pg.archipelago(16, algo=alg, prob=prob, pop_size=20)
#
# arch.evolve(10)
#
# arch.wait()
#
# results = [isl.get_population().champion_f[0] for isl in arch]
#
# print(results)
