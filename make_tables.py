import numpy    as np
import pandas   as pd
from glob       import glob
from tqdm       import tqdm

from utils      import load_data, make_tables
import dirs

targetError = 1e-8
numRuns     = 50

# Make all tables
for algorithm in ["ES", "ESMod"]:
    for dim in [10, 30]:
        make_tables(algorithm, dim, numRuns, targetError)
