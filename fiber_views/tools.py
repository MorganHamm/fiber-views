#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7 14:37:04 2022

@author: morgan

@description: A set of usefull tools for workign with fiber views
"""



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

from . import utils


# =============================================================================
# KMER COUNTING
# =============================================================================


def count_kmers(fiber_view, k):
    """
    Count k-mers in each fiber in a fiber view.

    This function counts the occurrences of k-mers in each fiber in a fiber view, and stores the resulting k-mer counts in the 'kmers' element of the `obsm` attribute of the fiber view. The length of the k-mers (`k`) and the mapping from k-mer strings to column indices in the k-mer count matrix are stored in the 'kmer_len' and 'kmer_idx' elements of the `uns` attribute, respectively.

    Parameters
    ----------
    fiber_view : anndata.AnnData
        Fiber view object containing DNA sequence data in the 'seq' element of the `layers` attribute.
    k : int
        Length of the k-mers to count.

    Returns
    -------
    None
        The function updates the fiber view object in place, adding a new observation matrix 'kmers' containing
        the counts of each k-mer for each fiber, and adds two new entries to the 'uns' dictionary: 'kmer_len' and
        'kmer_idx'. 'kmer_len' is the length of the k-mers that were counted, and 'kmer_idx' is a list of the
        k-mers that were counted, with each k-mer represented as a bytes object.
    """
    # count kmers in each fiber
    kmer_to_idx = {bytes(k) : v for (k, v) in \
                   zip(itertools.product(b'ACGT', repeat=k), range(4**k))}
    fiber_view.obsm['kmers'] = np.zeros((fiber_view.shape[0], 4**k))
    for m in range(fiber_view.shape[1] - k + 1):
        kmer_col = fiber_view.layers['seq'][:, m:m+k]
        for i, row in enumerate(kmer_col):
            fiber_view.obsm['kmers'][i, kmer_to_idx[bytes(row)]] += 1
    fiber_view.uns['kmer_len'] = k
    # kmer_idx is the column index of the kmer matrix
    fiber_view.uns['kmer_idx'] = list(kmer_to_idx.keys())
    return(None)


def calc_kmer_dist(fiber_view, metric='cityblock'):
    """
    Calculate pairwise k-mer distances between fibers in a fiber view.

    This function calculates pairwise distances between fibers in a fiber view based on the k-mer counts stored in the 'kmers' element of the `obsm` attribute. The distance metric can be specified using the `metric` parameter (default is 'cityblock'). The resulting distance matrix is stored in the 'kmer_dist' element of the `obsp` attribute of the fiber view.

    Parameters
    ----------
    fiber_view : anndata.AnnData
        Fiber view object containing k-mer count data in the 'kmers' element of the `obsm` attribute.
    metric : str, optional
        Distance metric to use for calculating pairwise distances. The default is 'cityblock'.

    Returns
    -------
    None
    """
    dists =  distance.pdist(fiber_view.obsm['kmers'], metric=metric)
    fiber_view.obsp['kmer_dist'] = distance.squareform(dists)
    return(None)




# =============================================================================
# PLOTTING
# =============================================================================


def plot_methylation(fiber_view, label_bases=False, ):
    """
    Plot a heatmap of DNA methylation levels in a fiber view.

    This function generates a heatmap of DNA methylation levels in a fiber view, with different colors representing different methylation states (unmethylated, m6A, and 5mC). The DNA sequence for each fiber can optionally be labeled on the plot by setting the `label_bases` parameter to `True`.

    Parameters
    ----------
    fiber_view : anndata.AnnData
        Fiber view object containing DNA methylation data in the 'm6a' and 'cpg' elements of the `layers` attribute, and DNA sequence data in the 'seq' element of the `layers` attribute.
    label_bases : bool, optional
        Whether to label each base in the DNA sequence on the plot. The default is `False`.

    Returns
    -------
    ax : matplotlib.axes._subplots.AxesSubplot
        Axes object for the plot.
    """
    mod_mtx = fiber_view.layers['m6a'].toarray() + \
        fiber_view.layers['cpg'].toarray() * 2
    mod_colors = ((0.65,    0.65,   0.65,   1.0), # grey, unmetthylated
                  (0.78,    0.243,  0.725,  1.0), # purple m6a
                  (0.78,    0.243,  0.243,  1.0)) # red, cpg
    cmap = LinearSegmentedColormap.from_list('Custom', mod_colors,
                                             len(mod_colors))
    if label_bases:
        ax = sns.heatmap(mod_mtx, cmap=cmap,
                         annot=fiber_view.layers['seq'].astype('U1'), fmt = '',
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
    # TODO: make this better
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

    return sns.lineplot(data=long_df, x='pos', y='value', hue='variable')

def simple_region_plot(fview, mod='m6a'):
    """
    Plot a heatmap of DNA methylation levels and DNA sequence features in a fiber view.

    This function generates a heatmap of DNA methylation levels and DNA sequence features in a fiber view. The data is displayed using a color map that indicates the presence or absence of different features (unmodified, nucleosome, m6A, m6A+nucleosome, msp, msp+m6A). The methylation state to be plotted can be specified using the `mod` parameter, which should be set to 'm6a' (default) or 'cpg'.

    Parameters
    ----------
    fview : anndata.AnnData
        Fiber view object containing DNA methylation data in the 'm6a' and 'cpg' elements of the `layers` attribute, and DNA sequence data in the 'seq' element of the `layers` attribute.
    mod : str, optional
        DNA methylation state to plot. Should be set to 'm6a' (default) or 'cpg'.

    Returns
    -------
    ax : matplotlib.axes._subplots.AxesSubplot
        Axes object for the plot.
    """
    # -0.5  :   no region, m6A
    # 0     :   no region
    # 0.5   :   nucleosome, m6A
    # 1     :   nucleosome
    # 1.5   :   msp, m6A
    # 2     :   msp
    nucs = make_dense_regions(fview, base_name = 'nuc', report='score')
    msps = make_dense_regions(fview, base_name = 'msp', report='score')
    return sns.heatmap(nucs + msps *2 - fview.layers[mod] * 0.5 - (fview.layers['seq'] == b'-'),
                cmap=sns.color_palette("Paired", 7), vmin=-1, vmax=2)




# =============================================================================
# REGIONS
# =============================================================================


def make_region_df(fview, base_name = 'nuc', zero_pos='left'):
    """
    Create a dataframe containing the positions and lengths of regions in a fiber view.

    Parameters
    ----------
    fview : anndata.AnnData
        The fiber view object containing the base modification data.
    base_name : str, optional
        The name of the type of regions to bin. This should be one of 'nuc' (nucleosomes), or 'msp' (methylation sensitive patches).
        The default value is 'nuc'.
    zero_pos : str, optional
        The position to use as the zero point for the start positions of the base modifications. This should be one of
        'left', 'center', or 'right'. The default value is 'left'.

    Returns
    -------
    pandas.DataFrame
        A dataframe with columns 'row' (the fiber index), 'start' (the start position of the base modification),
        'length' (the length of the base modification), and 'score' (the score of the base modification).
    """
    # Convert the position, length, and score data for the specified base to sparse matrices in COOrdinate format
    pos_coo = fview.layers["{}_pos".format(base_name)].tocoo()
    len_coo = fview.layers["{}_len".format(base_name)].tocoo()
    score_coo = fview.layers["{}_score".format(base_name)].tocoo()
    # Calculate the start position of each base modification based on the specified zero position
    if zero_pos == 'left':
        # the left edge of the fiber view window is considered 0
        start = pos_coo.col * fview.uns['bin_width'] - pos_coo.data
    elif zero_pos == 'center':
        # the 'center' of the fiber view window is considered 0
        start = fview.var.pos[pos_coo.col] - pos_coo.data
    else:
        # The right edge of the fiber view window is considered 0
        start = fview.var.pos[pos_coo.col] - pos_coo.data
    region_df = pd.DataFrame({
        'row' : pos_coo.row,
        'start' : start,
        'length' : len_coo.data,
        'score' : score_coo.data
        })
    return(region_df.drop_duplicates(ignore_index=True))

def make_dense_regions(fview, base_name = 'nuc', report="score"):
    """
    Create a dense matrix containing a representation of region infromation in a fiber view.

    Parameters
    ----------
    fview : anndata.AnnData
        The fiber view object containing the base modification data.
    base_name : str, optional
        The name of the type of regions to bin. This should be one of 'nuc' (nucleosomes), or 'msp' (methylation sensitive patches).
        The default value is 'nuc'.
    report : str, optional
        The data to include in the dense matrix. This should be one of 'score' or 'length'. The default value is 'score'.

    Returns
    -------
    numpy.ndarray
        A dense matrix of size (number of fibers, number of bases) containing the specified region data.
        Each position in the matrix where a region is not present is set to 0, positions where ar region is present
        may be set to either the length or score value of the region occupying that position.
    """
    region_df = make_region_df(fview, base_name=base_name)
    dense_mtx = np.zeros(fview.shape, dtype=region_df[report].dtype)
    for i, region in region_df.iterrows():
        start = max(region.start, 0)
        end = min(max(region.start + region.length, 0), dense_mtx.shape[1])
        dense_mtx[region.row, start:end] = region[report]
    return(dense_mtx)


def filter_regions(fview, base_name = 'nuc', length_limits = (-np.inf, np.inf),
                   score_limits = (-np.inf, np.inf), inplace=False):
    """
    Filter base modifications in a fiber view by length and score limits.

    Parameters
    ----------
    fview : anndata.AnnData
        The fiber view object containing the base modification data.
    base_name : str, optional
        The name of the type of regions to bin. This should be one of 'nuc' (nucleosomes), or 'msp' (methylation sensitive patches).
        The default value is 'nuc'.
    length_limits : tuple of float, optional
        The lower and upper limits for the length of the base modifications. Modifications with lengths outside of these
        limits will be filtered out. The default value is (-inf, inf), which includes all modifications.
    score_limits : tuple of float, optional
        The lower and upper limits for the score of the base modifications. Modifications with scores outside of these
        limits will be filtered out. The default value is (-inf, inf), which includes all modifications.
    inplace : bool, optional
        If True, the function will filter the base modifications in place and return None. If False (default), the
        function will return a new fiber view.

    Returns
    -------
    None or anndata.AnnData
        The function updates the fiber view object in place or returns a new fiber
        view with the selected region type filtered.
    """
    # filters regions with base_name by length and score limits
    if inplace:
        fview_out = fview
    else:
        fview_out = fview.copy()
    region_df = make_region_df(fview, base_name=base_name)
    region_df = region_df[(region_df.length > length_limits[0]) &
                          (region_df.length < length_limits[1]) &
                          (region_df.score > score_limits[0]) &
                          (region_df.score < score_limits[1])]
    pos_array, len_array, score_array = utils.make_sparse_regions(region_df, fview.shape)
    fview_out.layers["{}_pos".format(base_name)] = pos_array.tocsr()
    fview_out.layers["{}_len".format(base_name)] = len_array.tocsr()
    fview_out.layers["{}_score".format(base_name)] = score_array.tocsr()
    if inplace:
        return(None)
    else:
        return(fview_out)


def bin_sparse_regions(fview, base_name = 'nuc', bin_width = 10, interval = 3):
    """
    Bin regions in a fiber view by averaging their length and score over a set of consecutive bins.

    Parameters
    ----------
    fview : anndata.AnnData
        The fiber view object containing the region data.
    base_name : str, optional
        The name of the type of regions to bin. This should be one of 'nuc' (nucleosomes), or 'msp' (methylation sensitive patches).
        The default value is 'nuc'.
    bin_width : int, optional
        The width of each bin, in base pairs. The default value is 10.
    interval : int, optional
        The interval between bins, in base pairs. The default value is 3.

    Returns
    -------
    tuple of scipy.sparse.coo_matrix
        A tuple containing the position, length, and score data for the binned regions, stored as
        COOrdinate format sparse matrices.
    """
    region_df = make_region_df(fview, base_name=base_name, zero_pos='left')
    results = utils.make_sparse_regions(region_df, fview.shape,
                                          bin_width = bin_width,
                                          interval = interval)
    # returns coo matrices for pos, len, and score
    return(results)


def agg_by_obs_and_bin(fview, obs_group_var='site_name', bin_width=10,
                       obs_to_keep=['seqid', 'pos', 'strand', ''], fast=True):
    """
    Aggregate fiber view data by a group variable in the `obs` dataframe and bin by `bin_widht` basepairs.

    Parameters
    ----------
    fview : anndata.AnnData
        The fiber view object containing the data to be aggregated.
    obs_group_var : str, optional
        The name of the `obs` group variable to use for aggregation. The default value is 'site_name'.
        If `obs_group_var` is set to None, the fiber view will not be aggregated by rows and the row ordering will
        be preserved.
    bin_width : int, optional
        The width of each bin, in base pairs. The default value is 10.
        If `bin_width is 1, the data will not be binned.
    obs_to_keep : list of str, optional
        A list of observation metadata columns to keep in the aggregated data. The default value is ['seqid', 'pos', 'strand', ''].
    fast : bool, optional
        If True, the modification matrices will be converted to dense matrices for faster calculations. The default value is True.
        This may use more memory for large fiber view objects.

    Returns
    -------
    anndata.AnnData
        An aggregated version of the input fiber view object, with observations grouped and binned according to the specified parameters.
    """
    # obs_to_keep should be a list of column names, should not be read specific
    # if fast is True, mod matrices will be converted to dense for calculation
    t_start = time.time()
    if obs_group_var is None:
        # if None is passed to obs_group_var, process each row individually
        obs_group_var = 'row'
        new_obs = fview.obs.copy()
        new_obs['row'] = new_obs.index
        fview.obs['row'] = fview.obs.index # necessary for getting right row
        new_obs['n_seqs'] = 1
    else:
        if not obs_group_var in obs_to_keep:
            obs_to_keep.append(obs_group_var)
        obs_to_keep = [col  for col in obs_to_keep if col in fview.obs.columns]
        new_obs = fview.obs[obs_to_keep].groupby([obs_group_var]).first()
        new_obs['n_seqs'] = fview.obs.groupby([obs_group_var]).count().iloc[:,1]
    # the value in var.pos is the pos at the start of each window.
    new_var = pd.DataFrame({"pos" : list(range(fview.var.pos[0],
                                               fview.var.pos[fview.shape[1]-1]+1,
                                                             bin_width))})
    # create new AnnData object and populate uns data
    new_adata = ad.AnnData(obs=new_obs, var=new_var)
    new_adata.X = csr_matrix(new_adata.shape)
    new_adata.uns = fview.uns.copy()
    new_adata.uns['is_agg'] = True
    new_adata.uns['bin_width'] = bin_width
    del(new_adata.uns['region_report_interval'])
    # count occurence of each base in each bin
    print("bases_and_mods")

    # initialize layers
    for base in [b'A', b'C', b'G', b'T']:
        layer_name = "{}_count".format(base.decode())
        new_adata.layers[layer_name] = np.zeros(new_adata.shape, dtype=int)

    mod_matrices = {}
    for mod in fview.uns['mods']:
        if fast:
            mod_matrices[mod] = np.array(fview.layers[mod].toarray())
        else:
            mod_matrices[mod] = fview.layers[mod]
        layer_name = "{}_count".format(mod)
        new_adata.layers[layer_name] = np.zeros(new_adata.shape, dtype=int)
    layer_name = "cpg_site_count"
    new_adata.layers[layer_name] = np.zeros(new_adata.shape, dtype=int)
    # agg bases and mods
    for i, group in enumerate(list(new_adata.obs.index)):
        for j in np.arange(new_adata.shape[1]):
            rows = fview.obs[obs_group_var] == group
            bin_start = j*bin_width
            bin_end = (j+1)*bin_width
            for base in [b'A', b'C', b'G', b'T']:
                layer_name = "{}_count".format(base.decode())
                new_adata.layers[layer_name][i, j] = \
                    np.sum(fview.layers['seq'][rows, bin_start:bin_end] == base)
            for mod in fview.uns['mods']:
                 layer_name = "{}_count".format(mod)
                 new_adata.layers[layer_name][i, j] = \
                     np.sum(mod_matrices[mod][rows, bin_start:bin_end])
            layer_name = "cpg_site_count"
            new_adata.layers[layer_name][i, j] = \
                np.sum(fview.layers['cpg_sites'][rows, bin_start:bin_end])
    new_adata.layers['read_coverage'] = new_adata.layers['A_count'] + \
         new_adata.layers['C_count'] + new_adata.layers['G_count'] + \
         new_adata.layers['T_count']

    t_end = time.time()
    print(t_end - t_start)
    # aggregate regions
    print("regions")
    for region_type in fview.uns['region_base_names']:
        layer_name = "{}_coverage".format(region_type)
        new_adata.layers[layer_name] = np.zeros(new_adata.shape, dtype=float)
        region_df = make_region_df(fview, base_name=region_type, zero_pos='left')
        for m, region in region_df.iterrows():
            group = fview.obs[obs_group_var][region.row]
            i = new_adata.obs.index.get_loc(str(group))
            reg_bound_start = max(0, region.start)
            reg_bound_end = min(region.start + region.length, bin_width * new_adata.shape[1])
            reg_start_bin = reg_bound_start // bin_width
            reg_end_bin = reg_bound_end // bin_width
            if reg_start_bin == reg_end_bin:
                # if the region is fully contained in one bin
                new_adata.layers[layer_name][i, reg_start_bin] += \
                    region.score * (reg_bound_end - reg_bound_start)
                continue
            # bins fully covered by the region
            new_adata.layers[layer_name][i, reg_start_bin + 1:reg_end_bin] += \
                region.score * bin_width
            # partial bin at beginning of region
            new_adata.layers[layer_name][i, reg_start_bin] += \
                (bin_width - (reg_bound_start % bin_width)) * region.score
            # partial bin at end of region
            if reg_end_bin != new_adata.shape[1]:
                new_adata.layers[layer_name][i, reg_end_bin] += \
                    (reg_bound_end % bin_width) * region.score
    t_end = time.time()
    print(t_end - t_start)
    return(new_adata)



def get_sequences(fview):
    """Returns a list of strings where each string is the sequence of one row of the fview object.

    Parameters
    ----------
    fview : AnnData object
        The fiber view object containing the sequence data.

    Returns
    -------
    sequences : list
        A list of strings where each string is the sequence of one row of the fview object.
    """
    sequences = []
    for i in range(fview.shape[0]):
        sequences.append(fview.layers['seq'][i].tobytes().decode("ascii"))
    return sequences
