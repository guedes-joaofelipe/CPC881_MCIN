import numpy                as np
import pandas               as pd
import matplotlib.pyplot    as plt
from mpl_toolkits.mplot3d   import Axes3D

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

def plot_evolution(tablePath, save=False, fig_name="auto", show=True):
    '''
        Plot evolution of error over generations for given results table.
    '''
    fig = plt.figure(figsize=(24, 18))


    title = "Evolution of mean error over FES number for DE on F1"
    if fig_name == "auto":
        fig_name = title

    data = pd.read_excel(tablePath)
    # print(data)
    plt.semilogy(defs.fesScale, data["Mean"], 'k.', markersize='8', linestyle='-', linewidth='2', label='DE/best/1/bin')

    plt.xlim([-0.01, 1.0])
    # plt.ylim([0.0, 1.01])
    plt.xlabel('Percentage of MaxFES',fontsize= 'large')
    plt.ylabel('Mean Error',fontsize= 'large')
    plt.title(title)
    plt.legend(loc="upper right")

    if show is True:
        plt.show()
    if save is True:
        fig.savefig(dirs.figures+fig_name+".png",
                    orientation='portrait', bbox_inches='tight')


    return fig
