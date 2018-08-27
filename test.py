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
# alg = DifferentialEvolution(dim=dim, pop_size=popSize)
alg = GOParticleSwarmOptimizationSimple(func_id=1, dim=dim, pop_size=popSize)
# print(alg.population.shape)
print(alg.population)
# alg.mutation(alg.mutation_rand_1)
# alg.crossover_binomial(alg.mutatedPopulation)
# # print(alg.trialPopulation)
# alg.generation_jumping()

# index = 2
# targetVector = alg.population.iloc[index, :]
# for i in range(10):
#
# print(alg.bestVector)
alg.generation()
alg.generation()
# # print(alg.mutatedPopulation)
alg.generation()
print(alg.population)
# # # alg.crossover_binomial(alg.mutation())
# print(alg.population)

# tablePath = dirs.tables+"DE/"+"DE_table2_F14_dim10.xlsx"
# plot_evolution(tablePath, fig_name="auto", save=True)

# # TEST PYGMO ALGS
# import pygmo    as pg
#
# errorTol = 1e-08
# # funcList = [1, 2, 6, 7, 9, 14]   # Assignment function list
# funcList = [2]   # Assignment function list
# dimList  = [10]
# numRuns  = 50
#
# popSize  = 50
# param_F  = 0.9
# param_CR = 0.9
#
# numProcesses= os.cpu_count()-2
# # numEvolves = int(np.ceil(numRuns/numProcesses))
# print("numProcesses ", numProcesses)
#
# for dim in dimList:
#     maxFEvals = 10000*dim
#     # Compute upper bound for generation numbers
#     fEvalsGen = popSize
#     # maxGen    = maxFEvals/fEvalsGen
#     # print(maxGen)
#     maxGen    = int(maxFEvals/fEvalsGen)
#     # print(maxGen)
#     # maxGen    = 3500
#
#     for funcId in funcList:
#         print("\nFunction {}\nDim {}\nMax Generations {}\nNum Runs".format(funcId, dim, maxGen, numRuns))
#
#         fileName = "TEST_DE_F_{}_runs{}_dim{}".format(funcId, 24, dim)
#
#         prob = pg.problem(pg.cec2014(prob_id=funcId, dim=dim))
#
#         alg = pg.algorithm(pg.de(gen=maxGen, F=param_F, CR=param_CR, variant=2, xtol=errorTol, ftol=errorTol))
#
#         arch = pg.archipelago(numRuns, algo=alg, prob=prob, pop_size=popSize)
#
#         arch.evolve(1)
#
#         arch.wait()
#
#         results = [isl.get_population().get_f() for isl in arch]
#         # results = pd.DataFrame([isl.get_population().champion_f[0] for isl in arch])
#         print(np.shape(results))
#
#         # print(arch)
#         # print(arch[0].get_population())
#
#         results.to_hdf(dirs.results+fileName+".hdf", "Only")
#
# data = pd.read_hdf(dirs.results+fileName+".hdf")
# print(data)
# print(data.shape)
