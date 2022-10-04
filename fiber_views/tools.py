#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7 14:37:04 2022

@author: morgan

@description: A set of usefull tools for workign with fiber views
"""



import numpy as np
import pandas as pd


import os
import sys
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
    dists =  distance.pdist(fiber_view.obsm['kmers'], metric=metric)
    fiber_view.obsp['kmer_dist'] = distance.squareform(dists)
    return(None)




# =============================================================================
# PLOTTING
# =============================================================================


def plot_methylation(fiber_view, label_bases=False, ):
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

    sns.lineplot(data=long_df, x='pos', y='value', hue='variable')

def simple_region_plot(fview):
    # -0.5  :   no region, m6A
    # 0     :   no region
    # 0.5   :   nucleosome, m6A
    # 1     :   nucleosome
    # 1.5   :   msp, m6A
    # 2     :   msp 
    nucs = make_dense_regions(fview, base_name = 'nuc', report='score')
    msps = make_dense_regions(fview, base_name = 'msp', report='score')
    sns.heatmap(nucs + msps *2 - fview.layers['m6a'] * 0.5 - (fview.layers['seq'] == b'-'), 
                cmap=sns.color_palette("Paired", 7))
    



# =============================================================================
# REGIONS
# =============================================================================


def make_region_df(fview, base_name = 'nuc', zero_pos='left'):
    pos_coo = fview.layers["{}_pos".format(base_name)].tocoo()
    len_coo = fview.layers["{}_len".format(base_name)].tocoo()
    score_coo = fview.layers["{}_score".format(base_name)].tocoo()
    if zero_pos == 'left':
        # the right edge of the fiber view window is considered 0
        start = pos_coo.col * fview.uns['bin_width'] - pos_coo.data
    elif zero_pos == 'center':
        # the 'center' of the fiber view window is considered 0
        start = fview.var.pos[pos_coo.col] - pos_coo.data
    else:
        start = fview.var.pos[pos_coo.col] - pos_coo.data
    region_df = pd.DataFrame({
        'row' : pos_coo.row,
        'start' : start,
        'length' : len_coo.data,
        'score' : score_coo.data
        })
    return(region_df.drop_duplicates(ignore_index=True))
 
def make_dense_regions(fview, base_name = 'nuc', report="score"):
    region_df = make_region_df(fview, base_name=base_name)
    dense_mtx = np.zeros(fview.shape, dtype=region_df[report].dtype)
    for i, region in region_df.iterrows():
        start = max(region.start, 0)
        end = min(max(region.start + region.length, 0), dense_mtx.shape[1])
        dense_mtx[region.row, start:end] = region[report]
    return(dense_mtx)
 
    
def filter_regions(fview, base_name = 'nuc', length_limits = (-np.inf, np.inf), 
                   score_limits = (-np.inf, np.inf)):
    region_df = make_region_df(fview, base_name=base_name)
    region_df = region_df[(region_df.length > length_limits[0]) &
                          (region_df.length < length_limits[1]) &
                          (region_df.score > score_limits[0]) &
                          (region_df.score < score_limits[1])]
    pos_array, len_array, score_array = utils.make_sparse_regions(region_df, fview.shape)
    fview.layers["{}_pos".format(base_name)] = pos_array.tocsr()
    fview.layers["{}_len".format(base_name)] = len_array.tocsr()
    fview.layers["{}_score".format(base_name)] = score_array.tocsr()
    return(None)
     
    
def bin_sparse_regions(fview, base_name = 'nuc', bin_width = 10, interval = 3):
    region_df = make_region_df(fview, base_name=base_name, zero_pos='left')
    results = utils.make_sparse_regions(region_df, fview.shape, 
                                          bin_width = bin_width, 
                                          interval = interval)
    # returns coo matrices for pos, len, and score
    return(results)
    
def make_region_densities(fview, base_name = 'nuc'):
    # creates a dense 
    region_df = make_region_df(fview, base_name=base_name, zero_pos='left')
    density_mtx = np.zeros(fview.shape)
    pass
    
    
def agg_by_obs_and_bin(fview, obs_col_name='site_name', bin_width=10, cols_to_keep=[]):
    pass
    
    
    
    
    
    
    
    
    