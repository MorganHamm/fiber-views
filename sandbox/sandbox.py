#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  3 20:46:16 2022

@author: morgan
"""
import numpy as np
import pandas as pd

import os
import sys
import seaborn as sns
import matplotlib.pyplot as plt
import fiber_views as fv
import pysam



os.chdir(os.path.expanduser("~/git/fiber_views"))

bamfile = pysam.AlignmentFile("local/aligned.fiberseq.chr3_trunc.bam", "rb")

bed_data = fv.read_bed('local/TAIR10_genes.bed')
# bed_data.query('not chrom in ["chrC", "chrM"]', inplace=True)


anno_df = fv.bed_to_anno_df(bed_data)
anno_df.query('seqid == "chr3" & pos < 200000', inplace=True) 
fview = fv.FiberView(bamfile, anno_df)

sdata = fview.summarize_by_obs(cols_to_keep=list(anno_df.keys()))


# -----------------------------------------------------------------------------
# check that cpgs are landing on Cs...

fv2 = fview[fview.obs.gene_id == "AT3G01510"]

seq_array = fv2.layers['seq'].copy()

temp = seq_array[fv2.layers['cpg'].toarray()]
for i in range(100):
    print(temp[i])

temp = seq_array[fv2.layers['m6a'].toarray()]
for i in range(100):
    print(temp[i])


# -----------------------------------------------------------------------------
# Try plotting 



sdata.var['ATs'] = np.sum(sdata.layers['As'], axis=0).T + np.sum(sdata.layers['Ts'], axis=0).T

sdata.var['CpGs'] = np.sum(sdata.layers['CpGs'], axis=0).T

sdata.var['m6a'] = np.sum(sdata.layers['m6a'], axis=0).T

sdata.var['cpg'] = np.sum(sdata.layers['cpg'], axis=0).T

sdata.var['m6a_freq'] = sdata.var.m6a / sdata.var.ATs

sdata.var['cpg_freq'] = sdata.var.cpg / sdata.var.CpGs

plot_data = sdata.var.copy()
plot_data['bin'] = plot_data.pos // 10
plot_data = plot_data.groupby('bin').sum()
plot_data.cpg_freq = plot_data.cpg / plot_data.CpGs
plot_data.m6a_freq = plot_data.m6a / plot_data.ATs

sns.relplot(data=plot_data, kind='line', 
           x='bin', y='m6a_freq')

plt.show()

sns.relplot(data=plot_data, kind='line', 
           x='bin', y='cpg_freq')

plt.show()

# sdata.layers['cpg_freq'] = 

sns.heatmap(data=sdata.layers['m6a'])

# -----------------------------------------------------------------------------
# tile plot with bases
# slow, limit to ~40 x 100

fv2 = fview[fview.obs.gene_id == "AT3G01480", 200:400][30:70, ]

from matplotlib.colors import LinearSegmentedColormap

fv2.layers['mods'] = fv2.layers['m6a'].toarray() + fv2.layers['cpg'].toarray() * 2

myColors = ((0.65, 0.65, 0.65, 1.0), (0.78, 0.243, 0.725, 1.0), (0.78, 0.243, 0.243, 1.0))
cmap = LinearSegmentedColormap.from_list('Custom', myColors, len(myColors))

ax = sns.heatmap(fv2.layers['mods'], cmap=cmap, 
                 annot=fv2.layers['seq'].astype('U1'), fmt = '', 
                 annot_kws={'size' : 8})


colorbar = ax.collections[0].colorbar
colorbar.set_ticks([1./3, 1, 5./3])
colorbar.set_ticklabels(['NA', 'm6A', '5mC'])

# ax.hlines(np.arange(40), *ax.get_xlim(), colors=[(1.0, 1.0, 1.0)],
#           linewidths = 2.5)

ax.hlines(np.arange(40), 0, 300, colors=[(1.0, 1.0, 1.0)],
          linewidths = 2.5)


# -----------------------------------------------------------------------------
# tile plot no bases

fv2 = fview[fview.obs.gene_id == "AT3G01480", 200:1800]

from matplotlib.colors import LinearSegmentedColormap

fv2.layers['mods'] = fv2.layers['m6a'].toarray() + fv2.layers['cpg'].toarray() * 2

myColors = ((0.65, 0.65, 0.65, 1.0), (0.78, 0.243, 0.725, 1.0), (0.97, 0.097, 0.0, 1.0))
cmap = LinearSegmentedColormap.from_list('Custom', myColors, len(myColors))

ax = sns.heatmap(fv2.layers['mods'], cmap=cmap)


colorbar = ax.collections[0].colorbar
colorbar.set_ticks([1./3, 1, 5./3])
colorbar.set_ticklabels(['NA', 'm6A', '5mC'])

# ax.hlines(np.arange(40), *ax.get_xlim(), colors=[(1.0, 1.0, 1.0)],
#           linewidths = 2.5)

ax.hlines(np.arange(fv2.shape[0]), 0, fv2.shape[1]+5, colors=[(1.0, 1.0, 1.0)],
          linewidths = .5)
