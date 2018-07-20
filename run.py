import time
import numpy    as np
import pandas   as pd
from tqdm       import tqdm

from optimize   import optimize
from evolution  import EvolutionStrategyMod
import dirs

if __name__ == "__main__":
    # np.set_printoptions(precision=16)
    # np.set_printoptions(precision=4, floatmode='maxprec_equal', suppress=True)

    funcId      = 5
    dim         = 10
    numRuns     = 10
    successRate = 0
    targetError = 1e-8

    start = time.perf_counter()
    hist = pd.DataFrame()
    for i in tqdm(range(numRuns)):
        errorHist, fitnessHist = optimize(EvolutionStrategyMod, func_id=funcId, dim=dim, max_f_evals='auto', target_error=10e-8, verbose=True)

        bestError = errorHist.iloc[-1,:].min()
        errorHist["Run"] = np.ones(errorHist.shape[0], dtype=int)*i
        # print(errorHist)
        print(errorHist.shape)
        # bestError = errorHist.iloc[-1,:].drop(labels='Run', axis=1).min()

        hist = pd.concat([errorHist], ignore_index=True)

        if bestError <= targetError:
            successRate += 1

    elapsed = time.perf_counter() - start
    successRate = (successRate/numRuns)*100

    # print(hist)
    print("\nhist shape: ", hist.shape)

    print("\nElapsed time: {:.2f}s".format(elapsed) )
    print("Success rate: {:.2f}%".format(successRate))

    hist.to_hdf(dirs.resultsPath+"ES_func{}_runs{}_dim{}_succ_{:.2f}.hdf".format(funcId, numRuns, dim, successRate), "Only")
