import os
import time
import numpy    as np
import pandas   as pd
import pygmo    as pg
from glob       import glob
from tqdm       import tqdm
# import yarpiz   as yp

import dirs
from utils      import get_solution
from evolution   import ParticleSwarmOptimization

# dimList = [10, 30]
dimList = [10]
# funcList= [1, 2, 6, 7, 9, 14]   # Assignment function list
funcList= [2]

popSize = 50

numRuns = 3
verbose = True
successRate = 0
targetError = 1e-8

start = time.perf_counter()
for dim in dimList:
    maxFES   = 1e4*dim
    # maxFES   = 1000
    maxIters = int((maxFES - popSize)/popSize)
    for funcId in funcList:
        # FILENAME
        fileName = "PSO_F_{}_runs{}_dim{}".format(funcId, numRuns, dim)

        hist  = []
        error = []
        for runId in range(numRuns):
            fitnessHist = pd.DataFrame()
            errorHist = pd.DataFrame()

            alg = ParticleSwarmOptimization(funcId, pop_size = popSize, dim=dim, max_iters=maxIters)
            solution = get_solution(funcId, dim)

            fitnessHist = alg.all_generations()

            # Save error and fitness history
            errorHist   = fitnessHist.copy() - solution
            errorHist   = errorHist.apply(np.abs).reset_index(drop=True)
            errorHist["Run"] = np.ones(errorHist.shape[0], dtype=int)*runId

            # Mean and Best fitness values of last generation
            lastMeanFit = fitnessHist.iloc[-1, :].mean()
            lastBestFit = fitnessHist.iloc[-1, :].min()

            if verbose is True:
                print("\nMean Fitness: {:.4f}".format(lastMeanFit))
                print("Best Fitness: {:.4f}\n".format(lastBestFit))
                print("Solution: {:.4f}\nMean Diff: {:.4f}\nF Evals:   {}\n".format(solution, solution-lastMeanFit, alg.fitnessEvals))

            hist.append(errorHist)
            bestError = errorHist.drop(labels='Run', axis=1).min()
            error.append(bestError)
        # print(fitnessHist)
        # input()
        # print(errorHist)

        hist = pd.concat(hist, ignore_index=True)

        successRate = np.sum(np.where(np.less_equal(error, targetError), 1, 0))
        successRate = (successRate/numRuns)*100

        # print(hist)
        # input()

        # Save results
        hist.to_hdf(dirs.results+fileName+"_succ_{:.2f}.hdf".format(successRate), "Only")

    # Show elapsed time after all runs
    elapsed = time.perf_counter() - start
    print("\nElapsed time: {:.2f}s".format(elapsed))
