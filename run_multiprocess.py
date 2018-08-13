import time
import numpy            as np
import pandas           as pd
from tqdm               import tqdm
from multiprocessing    import Pool

from optimize   import optimize
from evolution  import EvolutionStrategyMod, DifferentialEvolution
import dirs


def aux_optim(run_id=0, func_id=5, dim=2, pop_size=30, max_f_evals='auto', target_error=10e-8):
    '''
        Auxiliary function for multiprocessing.
    '''
    np.random.seed()

    print("Run ID: ", run_id)
    errorHist, fitnessHist = optimize(DifferentialEvolution, func_id=func_id, dim=dim, pop_size=30, max_f_evals=max_f_evals,
                              target_error=target_error, verbose=True)

    errorHist["Run"] = np.ones(errorHist.shape[0], dtype=int)*run_id
    return errorHist, fitnessHist


if __name__ == "__main__":
    funcList = [2]
    # funcList = [1, 2, 6, 7, 9, 14]   # Assignment function list
    for funcId in funcList:
        print("\nFunction {:2d}\n".format(funcId))

        # Problem and Evaluation parameters
        dim         = 10
        numRuns     = 51
        popSize     = 50

        successRate = 0
        targetError = 1e-8
        max_f_evals = 'auto'

        numProcesses= 6

        start = time.perf_counter()
        hist = pd.DataFrame()
        with Pool(numProcesses) as p:
            # Build argument list
            # Arguments: numRuns x [runId, funcId, dim, max_f_evals, targetError]
            argList = []
            for runId in range(numRuns):
                argList.append([runId, funcId, dim, popSize, max_f_evals, targetError])

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
        hist.to_hdf(dirs.results+"ES_func{}_runs{}_dim{}_succ_{:.2f}.hdf".format(funcId, numRuns, dim, successRate), "Only")
