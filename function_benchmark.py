import numpy as np
import pandas as pd
import pygmo as pg

funcList= [1, 2, 6, 7, 9, 14]   # Assignment function list
dim = 10

x = np.zeros(dim)

print("\n")
for i in range(1, 31):
    prob = pg.problem(pg.cec2014(prob_id=i, dim=dim))
    print("f_{:2d}(0) = {:.6f}".format(i, prob.fitness(x)[0]))
