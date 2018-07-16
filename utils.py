import numpy    as np
import dirs

def get_solutions(func_list=[1], dim=10):
    '''
        func_list: List of function ids, between 1 and 31.
        dim      : Problem dimensionality

        Returns a solutions dictionary of global optima. Keys are function ids,
        values are corresponding global optima.
    '''
    import pygmo    as pg
    from glob       import glob

    solutions = dict()
    for i in func_list:
        if i < 23:
            prob = pg.problem(pg.cec2014(prob_id=i, dim=dim))
            shift_data = np.loadtxt(dirs.inputPath+"shift_data_{}.txt".format(i))

            solutions[i] = prob.fitness(shift_data[:dim])[0]

        if i >= 23:
            raise ValueError("f_{:2d} not yet implemented".format(i))
            return -1
    return solutions
