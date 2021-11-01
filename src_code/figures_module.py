import matplotlib.pyplot as plt
import matplotlib
import cycler
import numpy as np

cmap = matplotlib.cm.get_cmap('viridis_r')
colors = [cmap(n) for n in np.linspace(0, 1, 9)]

def prepare_standard_figure(nrows=1, ncols=1, sharex=False, sharey=False, width=3.375, aspect_ratio=1.61, tight=False, constrained=False):

    # plt.rc('text', usetex=True)
    # plt.rc('text.latex', preamble=r'\usepackage{amsmath},')
    # plt.rc('font', family='serif', serif=['Computer Modern'], size=7)
    # plt.rc('font', serif=['Computer Modern'], size=7)
    plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size': 7})
    plt.rc('text', usetex=True)
    plt.rc('lines', linewidth=1.0)
    plt.rc('xtick', labelsize='medium', direction="in", top=True)
    plt.rc('ytick', labelsize='medium', direction="in", right=True)
    plt.rc('legend', fontsize='small', numpoints=1, frameon=False, handlelength=1)
    plt.rc('axes', linewidth=0.5, labelsize='x-large')
    plt.rc('errorbar', capsize=1)
    plt.rc('savefig', dpi=300)

    # change color scheme
    #colors__ = ["#fff7ec", "#fee8c8", "#fdd49e", "#fdbb84", "#fc8d59", "#ef6548", "#d7301f", "#990000"]
    colors__ = ["#ffc30f", "#ff5733", "#c70039", "#900c3f", "#581845", 'k']
    plt.rcParams['axes.prop_cycle'] = cycler.cycler(color=colors__)

    fig_size = (width, width/aspect_ratio)
    f1, axs = plt.subplots(nrows=nrows, ncols=ncols, sharex=sharex, sharey=sharey, figsize=fig_size, tight_layout=tight, constrained_layout=constrained)
    return f1, axs


def set_share_axes(axs, target=None, sharex=False, sharey=False):
    if target is None:
        target = axs.flat[0]
    # Manage share using grouper objects
    for ax in axs.flat:
        if sharex:
            target._shared_x_axes.join(target, ax)
        if sharey:
            target._shared_y_axes.join(target, ax)
    # Turn off x tick labels and offset text for all but the bottom row
    if sharex and axs.ndim > 1:
        for ax in axs[:-1,:].flat:
            ax.xaxis.set_tick_params(which='both', labelbottom=False, labeltop=False)
            ax.xaxis.offsetText.set_visible(False)
    # Turn off y tick labels and offset text for all but the left most column
    if sharey and axs.ndim > 1:
        for ax in axs[:,1:].flat:
            ax.yaxis.set_tick_params(which='both', labelleft=False, labelright=False)
            ax.yaxis.offsetText.set_visible(False)
