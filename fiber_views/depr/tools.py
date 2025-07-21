
import numpy as np
import pandas as pd


import anndata as ad

import os
import sys
import time
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
import fiber_views as fv
import pysam

import itertools

from scipy.spatial import distance
from scipy.sparse import csr_matrix, coo_matrix, vstack

from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.cluster import hierarchy

from Bio.Align import MultipleSeqAlignment
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO

from matplotlib.colors import LinearSegmentedColormap

from fiber_views import utils




# =============================================================================
# PLOTTING
# =============================================================================


def plot_methylation(fview, label_bases=False, ):
    """
    Plot a heatmap of DNA methylation levels in a fiber view.

    This function generates a heatmap of DNA methylation levels in a fiber view, with different colors representing different methylation states (unmethylated, m6A, and 5mC). The DNA sequence for each fiber can optionally be labeled on the plot by setting the `label_bases` parameter to `True`.

    Parameters
    ----------
    fview : anndata.AnnData
        Fiber view object containing DNA methylation data in the 'm6a' and 'cpg' elements of the `layers` attribute, and DNA sequence data in the 'seq' element of the `layers` attribute.
    label_bases : bool, optional
        Whether to label each base in the DNA sequence on the plot. The default is `False`.

    Returns
    -------
    ax : matplotlib.axes._subplots.AxesSubplot
        Axes object for the plot.
    """
    mod_mtx = fview.layers['m6a'].toarray() + \
        fview.layers['cpg'].toarray() * 2
    mod_colors = ((0.65,    0.65,   0.65,   1.0), # grey, unmetthylated
                  (0.78,    0.243,  0.725,  1.0), # purple m6a
                  (0.78,    0.243,  0.243,  1.0)) # red, cpg
    cmap = LinearSegmentedColormap.from_list('Custom', mod_colors,
                                             len(mod_colors))
    if label_bases:
        ax = sns.heatmap(mod_mtx, cmap=cmap,
                         annot=fview.layers['seq'].astype('U1'), fmt = '',
                         annot_kws={'size' : 8})
    else:
        ax = sns.heatmap(mod_mtx, cmap=cmap)
    colorbar = ax.collections[0].colorbar
    colorbar.set_ticks([1./3, 1, 5./3])
    colorbar.set_ticklabels(['NA', 'm6A', '5mC'])
    ax.hlines(np.arange(mod_mtx.shape[0]), 0, mod_mtx.shape[1], colors=[(1.0, 1.0, 1.0)],
              linewidths = 1)
    return(ax)


def plot_summary(sdata, bin_width=10):
    """
    Plot a summary of DNA methylation levels and frequencies in a fiber view.

    This function generates a line plot of DNA methylation levels and frequencies in a fiber view, with different lines representing different methylation states (m6A and 5mC). The data is summarized by grouping it into bins of a specified width, which can be set using the `bin_width` parameter (default is 10). The plot shows the total number of m6A and 5mC sites, as well as the frequencies of these modifications within each bin.

    Parameters
    ----------
    sdata : anndata.AnnData
        Fiber view object containing DNA methylation data in the 'm6a' and 'cpg' elements of the `layers` attribute, and DNA sequence data in the 'seq' element of the `layers` attribute.
    bin_width : int, optional
        Width of the bins used to summarize the data. The default is 10.

    Returns
    -------
    ax : matplotlib.axes._subplots.AxesSubplot
        Axes object for the plot.
    """
    # TODO: remove this
    sdata.var['ATs'] = np.sum(sdata.layers['As'], axis=0).T + np.sum(sdata.layers['Ts'], axis=0).T
    sdata.var['CpGs'] = np.sum(sdata.layers['CpGs'], axis=0).T
    sdata.var['m6a'] = np.sum(sdata.layers['m6a'], axis=0).T
    sdata.var['cpg'] = np.sum(sdata.layers['cpg'], axis=0).T
    sdata.var['m6a_freq'] = sdata.var.m6a / sdata.var.ATs
    sdata.var['cpg_freq'] = sdata.var.cpg / sdata.var.CpGs

    plot_data = sdata.var.copy()
    plot_data['bin'] = plot_data.pos  // bin_width
    plot_data = plot_data.groupby('bin').sum()
    plot_data.pos = plot_data.pos / bin_width
    plot_data.cpg_freq = (plot_data.cpg / plot_data.CpGs) * 0.2
    plot_data.m6a_freq = plot_data.m6a / plot_data.ATs
    long_df = plot_data.melt(id_vars=['pos'])
    long_df = long_df.loc[(long_df['variable'] == 'm6a_freq') | (long_df['variable'] == 'cpg_freq')]

    sns.lineplot(data=long_df, x='pos', y='value', hue='variable')

def simple_region_plot(fview, mod='m6a', split_var=None):
    """
    Plot a heatmap of DNA methylation levels and DNA sequence features in a fiber view.

    This function generates a heatmap of DNA methylation levels and DNA sequence features in a fiber view. The data is displayed using a color map that indicates the presence or absence of different features (unmodified, nucleosome, m6A, m6A+nucleosome, msp, msp+m6A). The methylation state to be plotted can be specified using the `mod` parameter, which should be set to 'm6a' (default) or 'cpg'.

    Parameters
    ----------
    fview : anndata.AnnData
        Fiber view object containing DNA methylation data in the 'm6a' and 'cpg' elements of the `layers` attribute, and DNA sequence data in the 'seq' element of the `layers` attribute.
    mod : str, optional
        DNA methylation state to plot. Should be set to 'm6a' (default) or 'cpg'.
    split_var : str, optional
        The name of a field in fview.obs. if not None, a white line will be drawn between any rows where this value changes. Usefull for splitting clusters or groups of fibers up visually.

    Returns
    -------
    ax : matplotlib.axes._subplots.AxesSubplot
        Axes object for the plot.
    """
    # TODO remove this, use new plotting functions
    # -0.5  :   no region, m6A
    # 0     :   no region
    # 0.5   :   nucleosome, m6A
    # 1     :   nucleosome
    # 1.5   :   msp, m6A
    # 2     :   msp
    # colors = ['#DBDBDB','#aaaaaa','#4377d0','#9dfff9','#139a43','#b2eb76']
    # palette = sns.color_palette("Paired", 7)
    colors = ['#cfcfcf', '#000000', '#9b9b9b','#4377d0','#9dfff9','#e31a1c', '#fdbf6f']
    palette = sns.color_palette(colors, 7)
    nucs = make_dense_regions(fview, base_name = 'nuc', report='ones')
    msps = make_dense_regions(fview, base_name = 'msp', report='ones')
    hmap_data = pd.DataFrame(nucs + msps *2 - fview.layers[mod] * 0.5 - (fview.layers['seq'] == b'-'),
                             columns=fview.var['pos'])
    ax = sns.heatmap(hmap_data, cmap=palette, vmin=-1, vmax=2)
    if split_var != None:
        h_lines = []
        for i, group in enumerate(fview.obs[split_var]):
            if group != fview.obs[split_var][i-1]:
                h_lines.append(i)
        ax.hlines(h_lines, xmin=0, xmax=hmap_data.shape[1], color="white")

    
    cbar = ax.collections[0].colorbar
    d=3/7 # dist between colors
    cbar.set_ticks(np.arange(-1+d/2,2, d))
    cbar.set_ticklabels(['outside\nof read', 'methylation\nno region', 
                         'no region', 'methylation\nin nucleosome', 'nucleosome', 'methylation\nin MSP', 'MSP'])
    
    return ax
    

