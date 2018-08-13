import time
import numpy            as np
import pandas           as pd
from tqdm               import tqdm
from multiprocessing    import Pool

from optimize   import optimize
from evolution  import DifferentialEvolution
import dirs

dim     = 2
popSize = 5

alg = DifferentialEvolution(dim=dim, pop_size=popSize)
# print(alg.population.shape)
print(alg.population)

# index = 2
# targetVector = alg.population.iloc[index, :]
# for i in range(10):
alg.generation()
# # print(alg.population)
alg.generation()
# alg.crossover_binomial(alg.mutation())
print(alg.population)
