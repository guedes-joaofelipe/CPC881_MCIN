import numpy    as np
import dirs

def get_solutions(func_id=1, dim=10):
    '''
        func_list: List of function ids, between 1 and 31.
        dim      : Problem dimensionality

        Returns a solutions dictionary of global optima. Keys are function ids,
        values are corresponding global optima.
    '''
    import pygmo    as pg
    from glob       import glob

    if func_id < 23:
        prob = pg.problem(pg.cec2014(prob_id=func_id, dim=dim))
        shift_data = np.loadtxt(dirs.inputPath+"shift_data_{}.txt".format(func_id))

        solution = prob.fitness(shift_data[:dim])[0]

    if func_id >= 23:
        raise ValueError("f_{:2d} not yet implemented".format(func_id))
        return None
    return solution

def save_to_latex(result1_path, result2_path):
    import pandas   as pd

    data1 = pd.read_hdf(result1_path)
    # data2 = pd.read_hdf(result2_path)
    grouped = data1.groupby(by='Run')

    for group in grouped.groups:
        print(group)


    return data1
