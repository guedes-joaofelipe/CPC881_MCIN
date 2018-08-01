import numpy                as np
import matplotlib.pyplot    as plt
from mpl_toolkits.mplot3d   import Axes3D

import dirs

def plot_3d(compactDf, labels, save=True, show=False):
    '''
        Plot projection of first 3 principal components in a 3D plot.
    '''
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    posData = compactDf[labels == defs.posCode]
    negData = compactDf[labels == defs.negCode]

    ax.scatter(posData.iloc[:, 0], posData.iloc[:, 1], posData.iloc[:, 2], c='xkcd:fire engine red', alpha=0.3)
    ax.scatter(negData.iloc[:, 0], negData.iloc[:, 1], negData.iloc[:, 2], c='xkcd:twilight blue', alpha=0.3)

    ax.set_title("3D Scatter Plot")

    fig = plt.gcf()
    fig.set_size_inches(28, 28)
    plt.subplots_adjust(left=0.09, bottom=0.09, right=0.95, top=0.80,
    wspace=None, hspace=None)

    if show is True:
        plt.show()

    if save is True:
        # Save plots
        fig.savefig(dirs.figures+"Plot_3D"+".pdf", orientation='portrait', bbox_inches='tight')
        fig.savefig(dirs.figures+"Plot_3D"+".png", orientation='portrait', bbox_inches='tight')

    return fig, ax
