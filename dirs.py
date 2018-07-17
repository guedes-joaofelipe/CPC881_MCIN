inputPath   = "./input_data/"
resultsPath = "./results/"

import os

try:
    os.makedirs(resultsPath)
except OSError:
    pass
