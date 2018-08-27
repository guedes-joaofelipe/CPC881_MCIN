import numpy                as np
import pandas               as pd
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
    fig = plt.figure(figsize=(24, 18))

    title = fig_name

    # plt.xlim([-0.01, 1.01])
    # plt.ylim([0.0, 1.01])
    # plt.xlabel('Percentage of MaxFES',fontsize= 'large')
    plt.xlabel('Generations',fontsize= 'large')
    plt.ylabel('Mean Error',fontsize= 'large')
    plt.title(title)
    print(paths)

    for path in tqdm(paths):
        print("Processing Function")
        fileName   = path.split("/")[-1].replace(".hdf", "")
        folderName = path.split("/")[-2].replace(".hdf", "")

        data = pd.read_hdf(path)
        numRuns = data["Run"].max()
        errorTable = make_function_table(data, numRuns)

        index = np.array(errorTable["Mean"].index)
        index = index/np.amax(index)

        plt.loglog(index, errorTable["Mean"], markersize='8', linestyle='-',
                        linewidth='2', label=folderName)
        plt.legend(loc="upper right")


    if show is True:
        plt.show()
    if save is True:
        fig.savefig(dirs.evolution_plots+fig_name.replace(" ", "_")+".png",
                        orientation='portrait', bbox_inches='tight')

    return fig
