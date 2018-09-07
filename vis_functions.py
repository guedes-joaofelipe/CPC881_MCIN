import numpy                as np
import pandas               as pd
import matplotlib
import matplotlib.pyplot    as plt
from mpl_toolkits.mplot3d   import Axes3D
from tqdm                   import tqdm

from utils import make_function_table
import dirs
import defs

def plot_3d(X, Y, Z, save=True, fig_name="Plot_3D", show=False):
    '''
        Plot 3D function.

        X, Y, Z:    3D function input and output data. Each matrix must be in (N, N) format, as a numpy.meshgrid() output.
        save:       Determines if the figure should be saved to file.
                    'png' saves figure in png format.
                    'pdf' saves figure in pdf format.
                    'all' or True saves figure in both png and pdf formats, creating two files.

        fig_name:   If save is True, fig_name will be used as the figure filename.
        show:       If True, displays resulting figure.

        Returns resulting figure and axes objects.
    '''

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.plot_surface(X, Y, Z, cmap='bone')

    ax.set_title("3D Scatter Plot")

    fig = plt.gcf()
    fig.set_size_inches(26, 26)
    # plt.subplots_adjust(left=0.09, bottom=0.09, right=0.95, top=0.80, wspace=None,
    #                     hspace=None)
    plt.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=1.0, wspace=None,
                        hspace=None)

    if show is True:
        plt.show()

    # Save plots
    if (save == 'png') or (save == 'all') or (save is True):
        fig.savefig(dirs.figures+fig_name+".png", orientation='portrait', bbox_inches='tight')
    if (save == 'pdf') or (save == 'all') or (save is True):
        fig.savefig(dirs.figures+fig_name+".pdf", orientation='portrait', bbox_inches='tight')

    return fig, ax

def plot_evolution(paths, save=False, fig_name="Error evolution plot", show=True):
    '''
        Plot evolution of error over generations for given results table.

        paths: List of results filepaths
    '''
    fig = plt.figure(figsize=(28, 18))

    SMALL_SIZE = 12
    MEDIUM_SIZE = 12
    BIGGER_SIZE = 16

    # matplotlib.rc('font', size=SMALL_SIZE)          # controls default text sizes
    matplotlib.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    matplotlib.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    matplotlib.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    matplotlib.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    matplotlib.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
    matplotlib.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    title = fig_name
    yMax = 1e11
    yMin = 1e-8

    plt.xlim([0.0, 1.0])
    # plt.ylim([yMin, yMax])
    plt.xlabel('Percentage of MaxFES', fontsize= 'x-large')
    # plt.xlabel('Generations', fontsize= 'x-large')
    plt.ylabel('Mean Error', fontsize= 'x-large')
    plt.title(title, fontsize= 'x-large')

    for path in paths:
        print(path)

    for path in tqdm(paths):
        print("\nProcessing File: {}\n".format(path))
        # fileName   = path.split("/")[-1].replace(".hdf", "")
        folderName = path.split("/")[-2].replace(".hdf", "")

        data = pd.read_hdf(path)
        numRuns = data["Run"].max()
        errorTable = make_function_table(data, numRuns)

        index = np.array(errorTable["Mean"].index)
        index = index/np.amax(index)

        plt.plot(index, errorTable["Mean"], markersize='8', linestyle='-',
                        linewidth='2', label=folderName)

    # Plot a dashed line x = 1e-8
    # line = np.ones(index.shape[0])*1e-8
    # plt.semilogy(index, line, '--k', markersize='8', label='Target error')
    plt.legend(loc="best")

    if show is True:
        plt.show()
    if save is True:
        fig.savefig(dirs.evolution_plots+fig_name.replace(" ", "_")+".png",
                        orientation='portrait', bbox_inches='tight')

    return fig
