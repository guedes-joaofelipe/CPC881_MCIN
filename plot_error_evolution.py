import pygmo        as pg
import numpy        as np

from glob           import glob
from vis_functions  import plot_evolution
import dirs

targetFolder = dirs.results+"to_plot/"
funcList = [1, 2, 6, 7, 9, 14]   # Assignment function list
# funcList = [1, 6, 7, 9, 14]   # Assignment function list
# funcList = [2]   # Assignment function list
dimList = [10, 30]
figName = "Error evolution Function "

for dim in dimList:
    for funcId in funcList:
        print("\nDim {} Function {}".format(dim, funcId))
        pathList = glob(targetFolder+"**/*F_{}_*dim{}*.hdf".format(funcId, dim), recursive=True)

        for i in range(len(pathList)):
            pathList[i] = pathList[i].replace("\\", "/")

        plot_evolution(pathList, save=True, fig_name=figName+str(funcId)+" dim "+str(dim), show=False)
