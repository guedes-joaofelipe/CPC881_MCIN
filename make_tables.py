import numpy    as np
import pandas   as pd
from glob       import glob
from tqdm       import tqdm

from utils      import load_data, make_tables
import dirs

targetError = 1e-8
numRuns     = 51

# Make all tables
# algList = ["ES", "ESMod", "DE"]
algList = ["DE"]
for algorithm in algList:
    for dim in [10, 30]:
        make_tables(algorithm, dim, numRuns, targetError)
