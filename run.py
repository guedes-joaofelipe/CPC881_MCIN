import numpy    as np
import pandas   as pd
import pygmo    as pg
from glob       import glob
from tqdm       import tqdm

import dirs
from evolution  import EvolutionStrategy
from utils      import get_solutions

# np.set_printoptions(precision=16)
# np.set_printoptions(precision=4, floatmode='maxprec_equal', suppress=True)

funcList       = [1, 2, 6, 7, 9, 14]   # Assignment function list
funcId         = 1
dim            = 2
numGenerations = 200
popSize        = 50

es = EvolutionStrategy(dim=dim, func_id=funcId, pop_size=popSize)

fitnessHist = pd.DataFrame(np.zeros((popSize, numGenerations)))
print("")
for i in tqdm(range(numGenerations)):
    es.generation()
    pop = es.population

    # Save fitness history
    fitnessHist.loc[i,:] = pop["Fitness"]

    # meanFitness = pop["Fitness"].mean()
    # print("Generation {:4d}| Mean Fitness: {:12.4f}".format(i+1, meanFitness))

lastMeanFit = fitnessHist.iloc[-1, :].mean()
solution = get_solutions([funcId], dim)

print("\nMean Fitness: {:.4f}".format(lastMeanFit))
print("Solution: {:.4f}\nDiff    : {:.4f}".format(solution[funcId], solution[funcId]-lastMeanFit))
