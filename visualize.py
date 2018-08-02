import pygmo        as pg
import numpy        as np

import matplotlib.pyplot    as plt
from mpl_toolkits.mplot3d   import Axes3D


from vis_functions  import plot_3d
import dirs

dim         = 2
functionId  = 16
infLim      = -100
supLim      = +100
numPoints   = 1000


x = np.linspace(infLim, supLim, num=numPoints)
y = np.linspace(infLim, supLim, num=numPoints)
X, Y = np.meshgrid(x, y)

# for functionId in range(1, 22):
prob = pg.problem(pg.cec2014(prob_id=functionId, dim=dim))
Z = np.zeros((len(X), len(Y)))
for i in range(len(X)):
    for j in range(len(Y)):
        Z[i,j] = prob.fitness([X[i,j], Y[i,j]])

plot_3d(X, Y, Z, save='png', fig_name="Function {} 3D Plot".format(functionId), show=False)
