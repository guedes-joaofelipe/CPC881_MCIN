import os
import time
import numpy    as np
import pandas   as pd
import pygmo    as pg
from glob       import glob
from tqdm       import tqdm
from multiprocessing    import Pool

# import yarpiz   as yp

import dirs
from utils      import get_solution
from evolution   import ParticleSwarmOptimization

def aux_optim(run_id=0, func_id=5, dim=2, pop_size=30, max_iters=100):
    '''
        Auxiliary function for multiprocessing.
    '''
    np.random.seed()
    print("Run ID: ", run_id)
    fitnessHist = pd.DataFrame()
    errorHist = pd.DataFrame()

    alg = ParticleSwarmOptimization(func_id, pop_size = pop_size, dim=dim, max_iters=max_iters)
    solution = get_solution(func_id, dim)
    verbose = True

    fitnessHist = alg.all_generations()

    # Save error and fitness history
    errorHist   = fitnessHist.copy() - solution
    errorHist   = errorHist.apply(np.abs).reset_index(drop=True)
    errorHist["Run"] = np.ones(errorHist.shape[0], dtype=int)*run_id

    # Mean and Best fitness values of last generation
    lastMeanFit = fitnessHist.iloc[-1, :].mean()
    lastBestFit = fitnessHist.iloc[-1, :].min()

    if verbose is True:
        print("\nMean Fitness: {:.4f}".format(lastMeanFit))
        print("Best Fitness: {:.4f}".format(lastBestFit))
        print("Solution: {:.4f}\nMean Diff: {:.4f}\nF Evals:   {}\n".format(solution, solution-lastMeanFit, alg.fitnessEvals))


    return errorHist, fitnessHist

if __name__ == "__main__":
    dimList = [10, 30]
    # dimList = [30]
    funcList= [1, 2, 6, 7, 9, 14]   # Assignment function list
    # funcList= [1]

    popSize = 20

    numRuns = 51
    successRate = 0
    targetError = 1e-8

    numProcesses= os.cpu_count()-1
    print("Using {} cores.".format(numProcesses))

    start = time.perf_counter()
    for dim in dimList:
        maxFES   = 1e4*dim
        # maxFES   = 1000
        maxIters = int((maxFES - popSize)/popSize)
        for funcId in funcList:
            # FILENAME
            fileName = "PSO_F_{}_runs{}_dim{}".format(funcId, numRuns, dim)

            print("\nFunction {:2d}\n".format(funcId))

            hist  = []
            error = []
            # alg = ParticleSwarmOptimization(funcId, pop_size = popSize, dim=dim, max_iters=maxIters)
            # solution = get_solution(funcId, dim)
            with Pool(numProcesses) as p:
                # Build argument list
                argList = []
                for runId in range(numRuns):
                    argList.append([runId, funcId, dim, popSize, maxIters])

                for errorHist, fitnessHist in p.starmap(aux_optim, argList, chunksize=1):
                    hist.append(errorHist)

                    bestError = errorHist.drop(labels='Run', axis=1).min()
                    error.append(bestError)

                hist.append(errorHist)
                bestError = errorHist.drop(labels='Run', axis=1).min()
                error.append(bestError)

            hist = pd.concat(hist, ignore_index=True)

            successRate = np.sum(np.where(np.less_equal(error, targetError), 1, 0))
            successRate = (successRate/numRuns)*100

            # Save results
            hist.to_hdf(dirs.results+fileName+"_succ_{:.2f}.hdf".format(successRate), "Only")

        # Show elapsed time after all runs
        elapsed = time.perf_counter() - start
        print("\nElapsed time: {:.2f}s".format(elapsed))
