
# A matplotlib style based on the gala package by @adrn:
# github.com/adrn/gala

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic_2d
from matplotlib.colors import LogNorm


mpl_style = {

    # Lines
    'lines.linewidth': 1.7,
    'lines.antialiased': True,
    'lines.marker': '.',
    'lines.markersize': 5.,

    # Patches
    'patch.linewidth': 1.0,
    'patch.facecolor': '#348ABD',
    'patch.edgecolor': '#CCCCCC',
    'patch.antialiased': True,

    # images
    'image.origin': 'upper',

    # colormap
    'image.cmap': 'viridis',

    # Font
    'font.size': 12.0,
    'text.usetex': True,
    'text.latex.preamble': r'\usepackage{amsmath}',
    'text.latex.preview': True,
    'axes.unicode_minus': False,

    # Axes
    'axes.facecolor': '#FFFFFF',
    'axes.edgecolor': '#333333',
    'axes.linewidth': 1.0,
    'axes.grid': False,
    'axes.titlesize': 'x-large',
    'axes.labelsize': 'large',
    'axes.labelcolor': 'k',
    'axes.axisbelow': True,

    # Ticks
    'xtick.major.size': 8,
    'xtick.minor.size': 4,
    'xtick.major.pad': 6,
    'xtick.minor.pad': 6,
    'xtick.color': '#333333',
    'xtick.direction': 'in',
    'ytick.major.size': 8,
    'ytick.minor.size': 4,
    'ytick.major.pad': 6,
    'ytick.minor.pad': 6,
    'ytick.color': '#333333',
    'ytick.direction': 'in',
    'xtick.labelsize': 'medium',
    'ytick.labelsize': 'medium',

    # Legend
    'legend.fancybox': True,
    'legend.loc': 'best',

    # Figure
    'figure.figsize': [6, 6],
    'figure.facecolor': '1.0',
    'figure.edgecolor': '0.50',
    'figure.subplot.hspace': 0.5,

    # Other
    'savefig.dpi': 300,
}



def corner_hist(x, bins=30, label_names=None, show_ticks=False, **kwargs):
    N, D = x.shape
    A = D - 1
    fig, axes = plt.subplots(A, A, figsize=(2 * A, 2 * A))

    kwds = dict(cmap="Greys", norm=LogNorm())
    kwds.update(kwargs)
    
    for i, ax_row in enumerate(axes):
        for j, ax in enumerate(ax_row):
            if j >= i:
                ax.set_visible(False)
                continue

            H, xedges, yedges, binnumber = binned_statistic_2d(
                x.T[i], x.T[j], x.T[i], statistic="count", bins=bins)

            imshow_kwds = dict(
                aspect=np.ptp(xedges)/np.ptp(yedges), 
                extent=(xedges[0], xedges[-1], yedges[-1], yedges[0]))
            imshow_kwds.update(kwds)
            
            ax.imshow(H.T, **imshow_kwds)
            ax.set_ylim(ax.get_ylim()[::-1])
        

            if ax.is_last_row() and label_names is not None:
                ax.set_xlabel(label_names[j])
                
            if ax.is_first_col() and label_names is not None:
                ax.set_ylabel(label_names[i])
                
            if not show_ticks:
                ax.set_xticks([])
                ax.set_yticks([])
                
    fig.tight_layout()
    
    return fig


def corner_scatter(data, label_names=None, show_ticks=False, fig=None,
                   **kwargs):
    N, D = data.shape
    K = D 
    
    if fig is None:
        fig, axes = plt.subplots(K, K, figsize=(2 * K, 2 * K))
        
    else:
        axes = fig.axes
    
    kwds = dict(s=1, c="tab:blue", alpha=0.5)
    kwds.update(kwargs)
    
    axes = np.atleast_2d(axes).T
    
    for j, y in enumerate(data.T):
        for i, x in enumerate(data.T):
            
            try:
                ax = axes[K - i - 1, K - j - 1]
            
            except:
                continue
            
            if j >= i: 
                ax.set_visible(False)
                continue
            
            ax.scatter(x, y, **kwds)
            
            if not show_ticks:
                ax.set_xticks([])
                ax.set_yticks([])

            if ax.is_last_row() and label_names is not None:
                ax.set_xlabel(label_names[i])
                
            if ax.is_first_col() and label_names is not None:
                ax.set_ylabel(label_names[j])
                
    fig.tight_layout()
    
    return fig



def corner_scatter_compare(data_1, data_2, label_names=None, show_ticks=False, 
                           fig=None, scatter_kwds_1=None, scatter_kwds_2=None,
                           **kwargs):

    
    N, D = data_1.shape
    if data_1.shape != data_2.shape:
        raise ValueError("data_1 and data_2 must have the same shape")

    K = D 
    
    if fig is None:
        fig, axes = plt.subplots(K, K, figsize=(2 * K, 2 * K))
        
    else:
        axes = fig.axes
    
    common_kwds = dict(s=1, alpha=0.5)
    common_kwds.update(kwargs)

    d1_kwds = common_kwds.copy()
    d1_kwds.update(dict(c="#666666", zorder=1))
    d1_kwds.update(scatter_kwds_1 or dict())

    d2_kwds = common_kwds.copy()
    d2_kwds.update(dict(c="tab:blue", zorder=2))
    d2_kwds.update(scatter_kwds_2 or dict())

    axes = np.atleast_2d(axes).T
    
    for j in range(D):
        for i in range(D):
            try:
                ax = axes[K - i - 1, K - j - 1]
            
            except:
                continue
            
            if j >= i: 
                ax.set_visible(False)
                continue
            

            ax.scatter(data_1[:, i], data_1[:, j], **d1_kwds)
            ax.scatter(data_2[:, i], data_2[:, j], **d2_kwds)
            
            if not show_ticks:
                ax.set_xticks([])
                ax.set_yticks([])

            if ax.is_last_row() and label_names is not None:
                ax.set_xlabel(label_names[i])
                
            if ax.is_first_col() and label_names is not None:
                ax.set_ylabel(label_names[j])
                
    fig.tight_layout()
    
    return fig

