import os
import time
import numpy            as np
import pandas           as pd
from tqdm               import tqdm
from multiprocessing    import Pool

from optimize   import optimize
from utils      import make_tables
from evolution  import (DifferentialEvolution, DifferentialEvolutionSimple,
                        OppositionDifferentialEvolution, OppositionDifferentialEvolutionSimple,
                        ParticleSwarmOptimizationSimple, GOParticleSwarmOptimizationSimple)
import dirs


def aux_optim(algorithm, run_id=0, func_id=5, dim=2, pop_size=30, max_f_evals='auto', target_error=10e-8):
    '''
        Auxiliary function for multiprocessing.
    '''
    np.random.seed()

    print("Run ID: ", run_id)
    errorHist, fitnessHist = optimize(algorithm, func_id=func_id, dim=dim, pop_size=pop_size, max_f_evals=max_f_evals,
                              target_error=target_error, verbose=True)

    errorHist["Run"] = np.ones(errorHist.shape[0], dtype=int)*run_id
    return errorHist, fitnessHist


if __name__ == "__main__":
    # dimList  = [10, 30]
    dimList  = [30]
    funcList = [1, 2, 6, 7, 9, 14]   # Assignment function list
    # funcList = [2]

    # Problem and Evaluation parameters
    algorithm   = GOParticleSwarmOptimizationSimple
    numRuns     = 51
    popSize     = 40

    successRate = 0
    targetError = 1e-8
    max_f_evals = 'auto'
    # max_f_evals = 1000
    #TODO:  Pass parameters as a dictionary/json
    #       Save parameters in a file for reference

    numProcesses= os.cpu_count()-1
    print("Using {} cores.".format(numProcesses))

    start = time.perf_counter()
    for dim in dimList:
        for funcId in funcList:
            # FILENAME
            fileName = "SIMPLE_PSO_F_{}_runs{}_dim{}".format(funcId, numRuns, dim)

            print("\nFunction {:2d}\n".format(funcId))

            # hist = pd.DataFrame()
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

            successRate = (successRate/numRuns)*100

            print("\nhist shape: ", hist.shape)

            print("Success rate: {:.2f}%\n".format(successRate))

            # Save results
            hist.to_hdf(dirs.results+fileName+"_succ_{:.2f}.hdf".format(successRate), "Only")

    # Show elapsed time after all runs
    elapsed = time.perf_counter() - start
    print("\nElapsed time: {:.2f}s".format(elapsed))

    # After results are ready, format them into Excel tables
    algList = ["PSO"]
    for algorithm in algList:
        for dim in [10, 30]:
            make_tables(algorithm, dim, numRuns, targetError)
