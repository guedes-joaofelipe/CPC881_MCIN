# Define project paths
input           = "./input_data/"
results         = "./results/"
tables          = "./tables/"
figures         = "../figures/"
evolution_plots = figures+'evolution_plots/'

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

try:
    os.makedirs(evolution_plots)
except OSError:
    pass
