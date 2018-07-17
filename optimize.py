import numpy    as np
import pandas   as pd
import pygmo    as pg
from glob       import glob
from tqdm       import tqdm

import dirs
from evolution  import EvolutionStrategy
from utils      import get_solutions

def optimize(func_id=1, dim=2, max_f_evals='auto', target_error=10e-8, verbose=True):
    if max_f_evals == 'auto':
        max_f_evals = 10000*dim

    numGenerations = 200
    popSize        = 50

    es = EvolutionStrategy(dim=dim, func_id=func_id, pop_size=popSize)
    solution = get_solutions(func_id, dim)

    # Initialize variables
    fitnessHist = pd.DataFrame()
    error = 9999.0
    generation = 1
    print("")

    # do .. while loop
    while True:
        es.generation()
        pop = es.population
        # print(np.shape(pop["Fitness"]))

        # Save fitness history
        fitnessHist = fitnessHist.append(pop["Fitness"], sort=False)
        # print(np.shape(fitnessHist))
        # input()
        bestFitness = pop["Fitness"].min()
        error = np.abs(bestFitness - solution)

        # Stop Conditions
        if (es.fitnessEvals >= max_f_evals) or (error <= target_error):
            break

        generation += 1
    # print(generation)
    lastMeanFit = fitnessHist.iloc[generation-1, :].mean()
    lastBestFit = fitnessHist.iloc[generation-1, :].min()

    if verbose is True:
        print("\nMean Fitness: {:.4f}".format(lastMeanFit))
        print("Best Fitness: {:.4f}\n".format(lastBestFit))
        print("Solution: {:.4f}\nDiff    : {:.4f}\nF Evals:   {}".format(solution, solution-lastMeanFit, es.fitnessEvals))

    return fitnessHist, error
