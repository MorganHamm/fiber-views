#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  3 20:46:16 2022

@author: morgan
"""
import numpy as np
import pandas as pd

import os
import time
import seaborn as sns
import matplotlib.pyplot as plt
import fiber_views as fv
import pysam

import anndata as ad
from scipy.sparse import csr_matrix, coo_matrix, vstack

os.chdir(os.path.expanduser("~/git/fiber_views"))


# -----------------------------------------------------------------------------
# basic reading from bam file and summarizing 


bamfile = pysam.AlignmentFile("local/aligned.fiberseq.chr3_trunc.bam", "rb")
# bamfile = pysam.AlignmentFile("examples/data/chr3_sample.aligned.fiberseq.bam", "rb")

# bed_data = fv.read_bed('local/TAIR10_genes.bed')
bed_data = fv.read_bed('examples/data/TAIR10_genes.bed')
# bed_data.query('not chrom in ["chrC", "chrM"]', inplace=True)


anno_df = fv.bed_to_anno_df(bed_data)
anno_df = anno_df.query('seqid == "chr3" & pos < 100000') 

anno_df = anno_df.sample(5)


fview = fv.FiberView(bamfile, anno_df, window=(-2000, 4000), fully_span=False)

fview = fv.FiberView(bamfile, anno_df, window=(-2000, 2000), fully_span=True)


sdata = fv.tools.agg_by_obs_and_bin(fview, obs_group_var='site_name', bin_width=10, 
                                obs_to_keep=['seqid', 'pos', 'strand', 
                                             'gene_id', 'score'])

fv.tools.simple_region_plot(fview, mod='m6a', split_var='site_name')


# -----------------------------------------------------------------------------
# plotting

wd = 1

ax = fv.plot.make_plot_ax(fview)

# fv.plot.draw_fiber_lines(fview, ax)
fv.plot.draw_fiber_bars(fview, ax, width=wd)
fv.plot.draw_regions(fview, ax, color="orange", width=wd)
fv.plot.draw_split_lines(fview, ax, split_var="site_name")

tstart = time.time()
fv.plot.draw_mods_offset(fview, ax, mod='cpg_sites', color="blue", width=wd)
fv.plot.draw_mods_offset(fview, ax, mod='cpg', color="red", width=wd)
tend = time.time()
print(tend - tstart)


tstart = time.time()
fv.plot.draw_mods(fview, ax, mod='cpg_sites', color="blue", width=wd)
fv.plot.draw_mods(fview, ax, mod='cpg', color="red", width=wd)
tend = time.time()
print(tend - tstart)


# -----------------------------------------------------------------------------
# check that cpgs are landing on Cs...

# fv2 = fview[fview.obs.gene_id == "AT3G01510"]
fv2 = fview[1:100]

seq_array = fv2.layers['seq'].copy()

temp = seq_array[fv2.layers['cpg'].toarray()]
for i in range(100):
    print(temp[i])

temp = seq_array[fv2.layers['m6a'].toarray()]
for i in range(100):
    print(temp[i])

# debug for padding issue

fview = fv.FiberView(bamfile, anno_df.iloc[13:18,:], window=(-10000, 10000), fully_span=False)

fview = fv.FiberView(bamfile, anno_df.iloc[13:18,:], window=(-1500, 500), fully_span=False)

fv2 = fview[fview.obs.strand == "-"]
fv2 = fview[fview.obs.strand == "+"]

np.sum(fv2.layers['seq'] == b'-')

cpgs = fv2.layers['seq'][fv2.layers['cpg'].toarray()]
sum(cpgs == b'C')
sum(cpgs == b'A')
sum(cpgs == b'G')
sum(cpgs == b'T')
sum(cpgs == b'-')


mAs = fv2.layers['seq'][fv2.layers['m6a'].toarray()]
sum(mAs == b'A')
sum(mAs == b'C')
sum(mAs == b'G')
sum(mAs == b'T')
sum(mAs == b'-')


# -----------------------------------------------------------------------------
# making summary plots with percent methylated sites


sdata = fv.read_h5ad("local/all_genes_summary.h5ad")

sdata = sdata[sdata.obs.sort_values(by='score', ascending=False).index]

n_NAs = sum(sdata.obs.score.isnull())
split_point = len(sdata.obs) - n_NAs

decile_group_size = (split_point / 10) + 1

sdata.obs['decile'] = np.append(
    np.array(np.arange(split_point) // decile_group_size + 1, dtype=int),
    [0]*n_NAs)

sdata.obs['log_score'] = np.log10(sdata.obs.score + 1)
sdata.obs.log_score[split_point:] = 0


# sdata = sdata[~sdata.obs.score.isnull(), ]
# sdata.obs['log_score'] = np.log10(sdata.obs.score)
# sdata = sdata[sdata.obs.n_seqs < 500, ]

sns.histplot(data=sdata.obs, x='log_score')

fv.tools.plot_summary(sdata)

# run these 2 together
fv.tools.plot_summary(sdata[sdata.obs.log_score < 2]) # about lower 30%
fv.tools.plot_summary(sdata[sdata.obs.log_score > 2.95]) # about uper 30%

# -----------------------------------------------------------------------------
# testing regions (nucleosomes and msps)

fview = fv.FiberView(bamfile, anno_df.iloc[13:14,:], window=(-1500, 500), fully_span=False)

fv.tools.simple_region_plot(fview)

fv.tools.filter_regions(fview, base_name="msp", length_limits=(20, np.inf))



# -----------------------------------------------------------------------------
# binning regions


fview = fv.FiberView(bamfile, anno_df.iloc[13:14,:], window=(-1500, 500), fully_span=False)


bin_width = 10

nuc_pos_mtx, nuc_len_mtx, nuc_score_mtx = \
    fv.tools.bin_sparse_regions(fview, base_name='nuc', bin_width=bin_width)
msp_pos_mtx, msp_len_mtx, msp_score_mtx = \
    fv.tools.bin_sparse_regions(fview, base_name='msp', bin_width=bin_width)

new_adata = ad.AnnData(
            obs=fview.obs,
            var=pd.DataFrame({"pos" : np.arange(fview.var.pos[0], fview.var.pos[fview.shape[1]-1], bin_width)})
    )

new_adata.X = csr_matrix(new_adata.shape)
new_adata.uns = fview.uns
new_adata.uns['is_agg'] = True
new_adata.uns['bin_width'] = bin_width

new_adata.layers['nuc_pos'] = nuc_pos_mtx.tocsr()
new_adata.layers['nuc_len'] = nuc_len_mtx.tocsr()
new_adata.layers['nuc_score'] = nuc_score_mtx.tocsr()
new_adata.layers['msp_pos'] = msp_pos_mtx.tocsr()
new_adata.layers['msp_len'] = msp_len_mtx.tocsr()
new_adata.layers['msp_score'] = msp_score_mtx.tocsr()

new_adata.layers['m6a'] = csr_matrix(new_adata.shape)
new_adata.layers['seq'] = csr_matrix(new_adata.shape, dtype='S1')

temp2 = fv.tools.make_region_df(fview)
temp = fv.tools.make_region_df(new_adata)

# check that making region df from binned data results in same df.
np.sum(~(temp == temp2))



nucs = fv.tools.make_dense_regions(new_adata, base_name = 'nuc', report='score')
msps = fv.tools.make_dense_regions(new_adata, base_name = 'msp', report='score')
sns.heatmap(nucs + msps *2, cmap=sns.color_palette("Paired", 4))
    
sns.heatmap(msps, cmap=sns.color_palette("Paired", 4))

# -----------------------------------------------------------------------------
# Try aggregating with agg_by_obs_and_bin



fview = fv.FiberView(bamfile, anno_df, fully_span=False)



fv_agg = fv.tools.agg_by_obs_and_bin(fview, obs_group_var='site_name', bin_width=20, 
                            obs_to_keep=['seqid', 'pos', 'strand', 'gene_id', 'score'])

# sort by expression score
fv_agg = fv_agg[fv_agg.obs.sort_values(by='score', ascending=False).index]

# sns.heatmap(fv_agg.layers['msp_coverage'] / np.array(fv_agg.obs.n_seqs)[:, np.newaxis])
sns.heatmap(fv_agg.layers['msp_coverage'] / fv_agg.layers['read_coverage'])

sns.heatmap(fv_agg.layers['m6a_count'] / (fv_agg.layers['A_count'] + fv_agg.layers['T_count']))
            

fv_agg = fv_agg[fv_agg.obs.sort_values(by='score', ascending=False).index]            
            
temp = fv_agg.layers['msp_coverage'] / fv_agg.layers['read_coverage']

temp = fv_agg.layers['m6a_count'] / (fv_agg.layers['A_count'] + fv_agg.layers['T_count'])

temp2 = np.sum(temp, axis=0)

sns.scatterplot(x=fv_agg.var.pos, y=temp2)

sns.lineplot(x=fv_agg.var.pos, y=fv_agg.layers['read_coverage'][0,:]/10)


# -----------------------------------------------

fview = fv.FiberView(bamfile, anno_df.iloc[[13]], window=(-1500, 500), fully_span=False)


fv_agg = fv.tools.agg_by_obs_and_bin(fview, obs_group_var='site_name', bin_width=10, 
                            obs_to_keep=['seqid', 'pos', 'strand', 'gene_id', 'score'],
                            fast=True)

fv.tools.filter_regions(fview, base_name='msp', length_limits=(20, np.inf))


fv_agg2 = fv.tools.agg_by_obs_and_bin(fview, obs_group_var='read_name', bin_width=10, 
                            obs_to_keep=['seqid', 'pos', 'strand', 'gene_id', 'score'],
                            fast=True)


# -----------------------------------------------------------------------------
# testing max_reads

import time


t_start = time.time()
fview = fv.FiberView(bamfile, anno_df, fully_span=False)
t_end = time.time()

print("300_limit: {}".format(t_end - t_start))
# 300_limit: 45.291322231292725


t_start = time.time()
fview = fv.FiberView(bamfile, anno_df, fully_span=False, max_reads=100)
t_end = time.time()

print("100_limit: {}".format(t_end - t_start))
# 100_limit: 24.672104597091675
