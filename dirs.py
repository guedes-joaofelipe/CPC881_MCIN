# Define project paths
input   = "./input_data/"
results = "./results/"
tables  = "./tables/"
figures = "../figures/"

import os

# Create folders
try:
    os.makedirs(results)
except OSError:
    pass

try:
    os.makedirs(tables)
except OSError:
    pass

try:
    os.makedirs(figures)
except OSError:
    pass
