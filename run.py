import numpy    as np
import pandas   as pd
import pygmo    as pg
from glob       import glob

import dirs
from evolution  import EvolutionStrategy

# np.set_printoptions(precision=16)
np.set_printoptions(precision=4, floatmode='maxprec_equal', suppress=True)

funcList= [1, 2, 6, 7, 9, 14]   # Assignment function list
dim = 2

es = EvolutionStrategy(dim=dim, func_id=1, pop_size=5)

pop1 = es.population.copy()
print(pop1)
es.mutate()
pop2 = es.childrenPopulation.copy()
print(pop2)
print("\nDifference")
print(pop1 - pop2)
