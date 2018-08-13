import numpy    as np
import pandas   as pd
import pygmo    as pg
from glob       import glob
from tqdm       import tqdm

import dirs
from utils      import get_solution

def optimize(algorithm, func_id=1, dim=2, pop_size=30,  max_f_evals='auto', target_error=10e-8, verbose=True):
    if max_f_evals == 'auto':
        max_f_evals = 10000*dim

    alg = algorithm(dim=dim, func_id=func_id, pop_size=pop_size)
    solution = get_solution(func_id, dim)

    # Initialize variables
    fitnessHist = pd.DataFrame()
    errorHist = pd.DataFrame()
    generation = 1
    print("")
    # do .. while loop
    while True:
        alg.generation()
        pop = alg.population

        # Save error and fitness history
        fitnessHist = fitnessHist.append(pop["Fitness"], sort=False)
        errorHist   = fitnessHist.copy() - solution
        errorHist   = errorHist.apply(np.abs).reset_index(drop=True)

        bestError   = errorHist.iloc[-1,:].min()
        # Stop Conditions
        if (alg.fitnessEvals >= max_f_evals) or (bestError <= target_error):
            break

        generation += 1

    # Mean and Best fitness values of last generation
    lastMeanFit = fitnessHist.iloc[generation-1, :].mean()
    lastBestFit = fitnessHist.iloc[generation-1, :].min()

    if verbose is True:
        print("\nMean Fitness: {:.4f}".format(lastMeanFit))
        print("Best Fitness: {:.4f}\n".format(lastBestFit))
        print("Solution: {:.4f}\nDiff    : {:.4f}\nF Evals:   {}\n".format(solution, solution-lastMeanFit, alg.fitnessEvals))

    return errorHist, fitnessHist

if __name__ == "__main__":
    pass
