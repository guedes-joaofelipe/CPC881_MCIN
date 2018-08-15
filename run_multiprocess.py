import os
import time
import numpy            as np
import pandas           as pd
from tqdm               import tqdm
from multiprocessing    import Pool

from optimize   import optimize
from evolution  import (EvolutionStrategyMod, DifferentialEvolution,
                        OppositionDifferentialEvolution)
import dirs


def aux_optim(algorithm, run_id=0, func_id=5, dim=2, pop_size=30, max_f_evals='auto', target_error=10e-8):
    '''
        Auxiliary function for multiprocessing.
    '''
    np.random.seed()
    # algorithm = OppositionDifferentialEvolution

    print("Run ID: ", run_id)
    errorHist, fitnessHist = optimize(algorithm, func_id=func_id, dim=dim, pop_size=30, max_f_evals=max_f_evals,
                              target_error=target_error, verbose=True)

    errorHist["Run"] = np.ones(errorHist.shape[0], dtype=int)*run_id
    return errorHist, fitnessHist


if __name__ == "__main__":
    dimList  = [10]
    funcList = [1, 2, 6, 7, 9, 14]   # Assignment function list
    # funcList = [2]

    # Problem and Evaluation parameters
    algorithm   = OppositionDifferentialEvolution
    numRuns     = 51
    popSize     = 100

    successRate = 0
    targetError = 1e-8
    max_f_evals = 'auto'
    # max_f_evals = 1000
    #TODO: Save parameters in a txt file for reference

    numProcesses= os.cpu_count()-2

    for dim in dimList:
        for funcId in funcList:
            print("\nFunction {:2d}\n".format(funcId))

            start = time.perf_counter()
            hist = pd.DataFrame()
            with Pool(numProcesses) as p:
                # Build argument list
                # Arguments: numRuns x [runId, funcId, dim, max_f_evals, targetError]
                argList = []
                for runId in range(numRuns):
                    argList.append([algorithm, runId, funcId, dim, popSize, max_f_evals, targetError])

                hist = []
                error = []
                for errorHist, fitnessHist in p.starmap(aux_optim, argList, chunksize=1):
                    hist.append(errorHist)

                    bestError = errorHist.drop(labels='Run', axis=1).min()
                    error.append(bestError)

                hist = pd.concat(hist, ignore_index=True)
                successRate = np.sum(np.where(np.less_equal(error, targetError), 1, 0))

            elapsed = time.perf_counter() - start
            successRate = (successRate/numRuns)*100

            print("\nhist shape: ", hist.shape)

            print("\nElapsed time: {:.2f}s".format(elapsed) )
            print("Success rate: {:.2f}%\n".format(successRate))

            # Save results
            hist.to_hdf(dirs.results+"ODE_func{}_runs{}_dim{}_succ_{:.2f}.hdf".format(funcId, numRuns, dim, successRate), "Only")
