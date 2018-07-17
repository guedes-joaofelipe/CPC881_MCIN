from multiprocessing import Pool
import time
import numpy    as np
import pandas   as pd
from tqdm       import tqdm

from optimize   import optimize
import dirs


def aux_optim(run_id=0, func_id=5, dim=2, max_f_evals='auto', target_error=10e-8):
    np.random.seed()

    print("Run ID: ", run_id)
    errorHist, fitnessHist = optimize(func_id=func_id, dim=dim, max_f_evals=max_f_evals,
                              target_error=target_error, verbose=True)

    errorHist["Run"] = np.ones(errorHist.shape[0], dtype=int)*run_id
    return errorHist, fitnessHist


if __name__ == "__main__":
    # funcList = [1, 2, 6, 7, 9, 14]   # Assignment function list
    funcList = [2]
    for funcId in funcList:
        print("\nFunction {:2d}\n".format(funcId))

        # Problem and Evaluation parameters
        dim         = 10
        # funcId      = 5 # Shifted and Rotated Ackley Function
        numRuns     = 51
        successRate = 0
        targetError = 1e-8
        max_f_evals = 'auto'

        numProcesses= 4

        # if numRuns % numProcesses != 0:
        #     raise ValueError("NumRuns must be multiple of numProcesses")
        #     break

        start = time.perf_counter()
        hist = pd.DataFrame()
        with Pool(numProcesses) as p:
            # Build argument list
            # Arguments: numRuns x [runId, funcId, dim, max_f_evals, targetError]
            argList = []
            for runId in range(numRuns):
                argList.append([runId, funcId, dim, max_f_evals, targetError])

            hist = []
            error = []
            for errorHist, fitnessHist in p.starmap(aux_optim, argList, chunksize=1):
                hist.append(errorHist)

                bestError = errorHist.drop(labels='Run', axis=1).min()
                error.append(bestError)

            hist = pd.concat(hist, ignore_index=True)
            successRate = np.sum(np.where(np.less_equal(error, targetError), 1, 0))

            # p.close()

        elapsed = time.perf_counter() - start
        successRate = (successRate/numRuns)*100

        print("\nhist shape: ", hist.shape)

        print("\nElapsed time: {:.2f}s".format(elapsed) )
        print("Success rate: {:.2f}%\n".format(successRate))

        # Save results
        hist.to_hdf(dirs.resultsPath+"ES_func{}_runs{}_dim{}_succ_{:.2f}".format(funcId, numRuns, dim, successRate), "Only")
