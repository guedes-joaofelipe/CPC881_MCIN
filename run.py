import numpy    as np
import pandas   as pd
import pygmo    as pg
from glob       import glob

import dirs

np.set_printoptions(precision=16)

funcList= [1, 2, 6, 7, 9, 14]   # Assignment function list

dim = 50

x = np.zeros(dim)

print("\n")

def get_solutions(func_list=[1], dim=10):
    solutions = dict()
    for i in func_list:
        if i < 23:
            prob = pg.problem(pg.cec2014(prob_id=i, dim=dim))
            shift_data = np.loadtxt(dirs.inputPath+"shift_data_{}.txt".format(i))

            solutions[i] = prob.fitness(shift_data[:dim])[0]

            # print("f_{:2d}(0)  = {:.6f}".format(i, prob.fitness(x)[0]))
            # print("f_{:2d}(x*) = {:.6f}\n".format(i, prob.fitness(shift_data[:dim])[0]))

        if i >= 23:
            raise ValueError("f_{:2d} not yet implemented".format(i))
            return -1
    return solutions

sol = get_solutions(func_list=funcList, dim=10)

print(sol)
