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
import seaborn as sns
import matplotlib.pyplot as plt
import fiber_views as fv
import pysam

import itertools

from scipy.spatial import distance

from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.cluster import hierarchy

from Bio.Align import MultipleSeqAlignment
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO


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

from matplotlib.colors import LinearSegmentedColormap


def plot_methylation(fiber_view, label_bases=False, ):

    mod_mtx = fiber_view.layers['m6a'].toarray() + \
        fiber_view.layers['cpg'].toarray() * 2
    
    mod_colors = ((0.65,    0.65,   0.65,   1.0), # grey, unmetthylated
                  (0.78,    0.243,  0.725,  1.0), # purple m6a
                  (0.78,    0.243,  0.243,  1.0)) # red, cpg
    cmap = LinearSegmentedColormap.from_list('Custom', mod_colors, len(mod_colors))
    
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



