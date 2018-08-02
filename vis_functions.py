import numpy                as np
import matplotlib.pyplot    as plt
from mpl_toolkits.mplot3d   import Axes3D

import dirs

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

# def plot_evolution(X, Y, Z, population, save=True, fig_name="Evolution_3D", show=False):
#     '''
#         Plot evolution of population over target function surface.
#     '''
#
#     fig, ax = plot_3d(X, Y, Z, save=False, show=False)
#
#
#
#     return fig, ax
