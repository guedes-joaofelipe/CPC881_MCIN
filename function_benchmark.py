import numpy    as np
import pandas   as pd
import pygmo    as pg

import dirs

# funcList = [1, 2, 6, 7, 9, 14]   # Assignment function list
funcList = [ 23]
funcList = range(1, 31)
dim = 10

x = np.zeros(dim)

print("\n")
for i in funcList:
    if i < 23:
        prob = pg.problem(pg.cec2014(prob_id=i, dim=dim))
        shift_data = np.loadtxt(dirs.inputPath+"shift_data_{}.txt".format(i))

        print("f_{:2d}(0)  = {:.6f}".format(i, prob.fitness(x)[0]))
        print("f_{:2d}(x*) = {:.6f}\n".format(i, prob.fitness(shift_data[:dim])[0]))

    if i >= 23:
        print("f_{:2d} not yet implemented".format(i))
        # prob = pg.problem(pg.cec2014(prob_id=i, dim=dim))
        # shift_data = np.loadtxt(dirs.inputPath+"shift_data_{}.txt".format(i))
        #
        #
        # print("f_{:2d}(0)  = {:.6f}".format(i, prob.fitness(x)[0]))
        # print("f_{:2d}(x*) = {:.6f}\n".format(i, prob.fitness(shift_data[:dim*5])[0]))
