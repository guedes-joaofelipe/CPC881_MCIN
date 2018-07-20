# Define project paths
inputPath   = "./input_data/"
resultsPath = "./results/"
tablesPath  = "./tables/"

import os

# Create folders
try:
    os.makedirs(resultsPath)
except OSError:
    pass

try:
    os.makedirs(tablesPath)
except OSError:
    pass
