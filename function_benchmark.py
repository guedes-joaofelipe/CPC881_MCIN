""" Benchmark to test the cec2014 objective-functions 
"""

import numpy    as np
import pandas   as pd
import pygmo    as pg
import dirs

func_list = [1, 2, 6, 7, 9, 14]   # Assignment function list
dim = 10

# Initializing a test vector with zero elements
x = np.zeros(dim)

for func_id in func_list:
    if func_id < 23:
        prob = pg.problem(pg.cec2014(prob_id=func_id, dim=dim))
        shift_data = np.loadtxt(dirs.input+"shift_data_{}.txt".format(func_id))

        print("f_{:2d}(0)  = {:.6f}".format(func_id, prob.fitness(x)[0]))
        print("f_{:2d}(x*) = {:.6f}\n".format(func_id, prob.fitness(shift_data[:dim])[0]))
    else: 
        print("f_{:2d} not yet implemented".format(func_id))
