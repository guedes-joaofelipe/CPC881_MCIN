import time
import numpy    as np
import pandas   as pd
from tqdm       import tqdm

from optimize   import optimize
import dirs

# np.set_printoptions(precision=16)
# np.set_printoptions(precision=4, floatmode='maxprec_equal', suppress=True)

funcId      = 1
numRuns     = 10
successRate = 0
targetError = 10e-8

start = time.perf_counter()
hist = pd.DataFrame()
for i in tqdm(range(numRuns)):
    newHist, error = optimize(func_id=funcId, dim=2, max_f_evals='auto', target_error=10e-8, verbose=True)

    newHist["Run"] = np.ones(newHist.shape[0], dtype=int)*i

    hist = pd.concat([hist, newHist], ignore_index=True)

    if error <= targetError:
        successRate += 1

elapsed = time.perf_counter() - start

print("\nhist shape: ", hist.shape)
successRate = (successRate/numRuns)*100
print("\nElapsed time: {:.2f}s".format(elapsed) )
print("Success rate: {:.2f}%".format(successRate))
hist.to_hdf(dirs.resultsPath+"ES_func_{}_success_X".format(funcId), "ES_func1_dim{}_succ_{:.2f}".format(2, successRate))
