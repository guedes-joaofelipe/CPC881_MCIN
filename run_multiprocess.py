from multiprocessing import Pool
import time
import numpy    as np
import pandas   as pd
from tqdm       import tqdm

from optimize   import optimize
import dirs


# def f(x):
#     df = pd.DataFrame([x, x*x])
#     return df
#
# if __name__ == '__main__':
#     with Pool(5) as p:
#         result = p.map(f, [1, 2, 3])
#
#     for df in result:
#         print(df)

def aux_optim(run_id=0, func_id=5, dim=2, max_f_evals='auto', target_error=10e-8):
    print("Run ID: ", run_id)
    newHist, error = optimize(func_id=func_id, dim=dim, max_f_evals=max_f_evals, target_error=target_error, verbose=True)

    newHist["Run"] = np.ones(newHist.shape[0], dtype=int)*run_id

    return newHist, error


if __name__ == "__main__":
    # Problem and Evaluation parameters
    dim         = 2
    funcId      = 5 # Shifted and Rotated Ackley Function
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
        for newHist, newError in p.starmap(aux_optim, argList, chunksize=1):
            hist.append(newHist)
            error.append(newError)


        hist = pd.concat(hist, ignore_index=True)
        successRate = np.sum(np.where(np.less_equal(error, targetError), 1, 0))

    elapsed = time.perf_counter() - start

    print("\nhist shape: ", hist.shape)
    successRate = (successRate/numRuns)*100
    print("\nElapsed time: {:.2f}s".format(elapsed) )
    print("Success rate: {:.2f}%".format(successRate))
    hist.to_hdf(dirs.resultsPath+"ES_func_{}_success_X".format(funcId), "ES_func1_dim{}_succ_{:.2f}".format(2, successRate))
