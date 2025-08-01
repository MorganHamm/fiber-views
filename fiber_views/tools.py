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


def count_kmers(fview, k):
    """
    Count k-mers in each fiber in a fiber view.

    This function counts the occurrences of k-mers in each fiber in a fiber view, and stores the resulting k-mer counts in the 'kmers' element of the `obsm` attribute of the fiber view. The length of the k-mers (`k`) and the mapping from k-mer strings to column indices in the k-mer count matrix are stored in the 'kmer_len' and 'kmer_idx' elements of the `uns` attribute, respectively.

    Parameters
    ----------
    fview : anndata.AnnData
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
    fview.obsm['kmers'] = np.zeros((fview.shape[0], 4**k))
    for m in range(fview.shape[1] - k + 1):
        kmer_col = fview.layers['seq'][:, m:m+k]
        for i, row in enumerate(kmer_col):
            fview.obsm['kmers'][i, kmer_to_idx[bytes(row)]] += 1
    fview.uns['kmer_len'] = k
    # kmer_idx is the column index of the kmer matrix
    fview.uns['kmer_idx'] = list(kmer_to_idx.keys())
    return(None)


def calc_kmer_dist(fview, metric='cityblock'):
    """
    Calculate pairwise k-mer distances between fibers in a fiber view.

    This function calculates pairwise distances between fibers in a fiber view based on the k-mer counts stored in the 'kmers' element of the `obsm` attribute. The distance metric can be specified using the `metric` parameter (default is 'cityblock'). The resulting distance matrix is stored in the 'kmer_dist' element of the `obsp` attribute of the fiber view.

    Parameters
    ----------
    fview : anndata.AnnData
        Fiber view object containing k-mer count data in the 'kmers' element of the `obsm` attribute.
    metric : str, optional
        Distance metric to use for calculating pairwise distances. The default is 'cityblock'.

    Returns
    -------
    None
    """
    dists =  distance.pdist(fview.obsm['kmers'], metric=metric)
    fview.obsp['kmer_dist'] = distance.squareform(dists)
    return(None)

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

def make_dense_regions(fview, base_name = 'nuc', report="ones"):
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
        The data to include in the dense matrix. This should be one of 'ones', 'score' or 'length'. The default value is 'score'.

    Returns
    -------
    numpy.ndarray
        A dense matrix of size (number of fibers, number of bases) containing the specified region data.
        Each position in the matrix where a region is not present is set to 0, positions where ar region is present
        may be set to either the length or score value of the region occupying that position.
    """
    if report == 'ones':
        dtype = int
    else:
        dtype = region_df[report].dtype
    region_df = make_region_df(fview, base_name=base_name)
    dense_mtx = np.zeros(fview.shape, dtype=dtype)
    for i, region in region_df.iterrows():
        start = max(region.start, 0)
        end = min(max(region.start + region.length, 0), dense_mtx.shape[1])
        if report == 'ones':
            dense_mtx[region.row, start:end] = 1
        else:
            dense_mtx[region.row, start:end] = region[report]
    return(dense_mtx)

def filter_regions(fview, base_name = 'nuc', new_base_name = None, 
                   length_limits = (-np.inf, np.inf), 
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
    new_base_name : str, optional
        If not `None`, the new region base name to save the filtered regions to (region information at base_name will not be modified).
        If new_base_name is None, the filtered regions will be saved to base_name.
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
    if new_base_name == None:
        new_base_name = base_name
    elif new_base_name not in fview_out.uns['region_base_names']:
        fview_out.uns['region_base_names'].append(new_base_name)
    region_df = make_region_df(fview, base_name=base_name)
    region_df = region_df[(region_df.length > length_limits[0]) &
                          (region_df.length < length_limits[1]) &
                          (region_df.score > score_limits[0]) &
                          (region_df.score < score_limits[1])]
    pos_array, len_array, score_array = utils.make_sparse_regions(region_df, fview.shape, interval=fview.uns['region_report_interval'])
    fview_out.layers["{}_pos".format(new_base_name)] = pos_array.tocsr()
    fview_out.layers["{}_len".format(new_base_name)] = len_array.tocsr()
    fview_out.layers["{}_score".format(new_base_name)] = score_array.tocsr()
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

# ============================================================================
# AGGREGATION
# ============================================================================

def agg_by_obs_and_bin(fview, obs_group_var='site_name', bin_width=10,
                       obs_to_keep=['seqid', 'pos', 'strand', ''], fast=True,
                       region_weights = 'ones'):
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
    region_weights : str, optional
        how to weight regions when aggregating must be one of 'ones', 'length' or 'score'

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
        region_df['ones'] = 1
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
                    region[region_weights] * (reg_bound_end - reg_bound_start)
                continue
            # bins fully covered by the region
            new_adata.layers[layer_name][i, reg_start_bin + 1:reg_end_bin] += \
                region[region_weights] * bin_width
            # partial bin at beginning of region
            new_adata.layers[layer_name][i, reg_start_bin] += \
                (bin_width - (reg_bound_start % bin_width)) * region[region_weights]
            # partial bin at end of region
            if reg_end_bin != new_adata.shape[1]:
                new_adata.layers[layer_name][i, reg_end_bin] += \
                    (reg_bound_end % bin_width) * region[region_weights]
    t_end = time.time()
    print(t_end - t_start)
    return(new_adata)



# =============================================================================
# MISC
# =============================================================================

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


def get_seq_records(fview, id_col="read_name"):
    # TODO add doc string
    # TODO why is this different from get_sequences?
    seqs = [Seq(bytes(row)) for row in fview.layers['seq']]
    seq_records = []
    for i in range(fview.shape[0]):
        seq_records.append(SeqRecord(seqs[i], id=fview.obs[id_col][i], 
                                      description=fview.obs.index[i]))
    return(seq_records)



def mark_cpg_sites(fview, sparse=True):
    # make a new layer 'cpg_sites' with cpg sites as True, 
    # Known issue: all Cs at end of sequence are marked as not CpGs 
    # TODO add doc string
    cpg_sites = np.logical_and(fview.layers['seq'][:, 0:-1] == b'C', 
                               fview.layers['seq'][:, 1:] == b'G')
    cpg_sites = np.pad(cpg_sites, pad_width=((0,0),(0,1)), mode='constant')
    if sparse:
        cpg_sites = csr_matrix(cpg_sites)
    fview.layers['cpg_sites'] = cpg_sites
    if 'cpg_sites' not in fview.uns['mods']:
        fview.uns['mods'].append('cpg_sites')
    return(None)



def split_fire(fview, input_region='msp', threshold=1, output_regions=['lnk', 'fire']):
    # add linker and fire regions based on msp score
    # TODO add doc string
    filter_regions(fview, input_region, new_base_name=output_regions[0], 
                        score_limits=(-np.inf, threshold), inplace=True)
    filter_regions(fview, input_region, new_base_name=output_regions[1], 
                        score_limits=(threshold, np.inf), inplace=True)
    return(None)
    



