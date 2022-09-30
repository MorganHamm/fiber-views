#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  3 20:46:16 2022

@author: morgan
"""
import numpy as np
import pandas as pd

import os
import seaborn as sns
import matplotlib.pyplot as plt
import fiber_views as fv
import pysam

os.chdir(os.path.expanduser("~/git/fiber_views"))


# -----------------------------------------------------------------------------
# basic reading from bam file and summarizing 


bamfile = pysam.AlignmentFile("local/aligned.fiberseq.chr3_trunc.bam", "rb")

bed_data = fv.read_bed('local/TAIR10_genes.bed')
# bed_data.query('not chrom in ["chrC", "chrM"]', inplace=True)


anno_df = fv.bed_to_anno_df(bed_data)
anno_df.query('seqid == "chr3" & pos < 200000', inplace=True) 
fview = fv.FiberView(bamfile, anno_df, fully_span=False)


sdata = fview.summarize_by_obs(cols_to_keep=list(anno_df.keys()))


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



# -----------------------------------------------------------------------------
# making summary plots with percent methylated sites


sdata = fv.read_h5ad("local/all_genes_summary.h5ad")


sdata = sdata[~sdata.obs.score.isnull(), ]
sdata.obs['log_score'] = np.log10(sdata.obs.score)
sdata = sdata[sdata.obs.n_seqs < 500, ]

sns.histplot(data=sdata.obs, x='log_score')

fv.tools.plot_summary(sdata)

# run these 2 together
fv.tools.plot_summary(sdata[sdata.obs.log_score < 2]) # about lower 30%
fv.tools.plot_summary(sdata[sdata.obs.log_score > 2.95]) # about uper 30%

# -----------------------------------------------------------------------------
# testing regions (nucleosomes and msps)




fview = fv.FiberView(bamfile, anno_df.iloc[13:18,:], window=(-1500, 500), fully_span=True)

temp = fview.layers['nuc_pos'].toarray()
# sns.heatmap(temp)



temp2 = temp[:,np.arange(0,2000, 30)]
sns.heatmap(temp2)

    


# --------------



temp = fv.tools.make_dense_regions(fview, base_name = 'nuc', report='score')
temp[temp > 0] = np.minimum(temp[temp > 0] * 4, 1000)
sns.heatmap(temp)

nucs = fv.tools.make_dense_regions(fview, base_name = 'nuc', report='score')
msps = fv.tools.make_dense_regions(fview, base_name = 'msp', report='score')

sns.heatmap(nucs + msps *2 - fview.layers['m6a'] * 0.5, cmap=sns.color_palette("Set2", 6))

# -0.5  :   no region, m6A
# 0     :   no region
# 0.5   :   nucleosome, m6A
# 1     :   nucleosome
# 1.5   :   msp, m6A
# 2     :   msp 


sns.heatmap(nucs + msps *2 - fview.layers['m6a'] * 0.5 - (fview.layers['seq'] == b'-'), 
            cmap=sns.color_palette("Paired", 7))

temp = nucs + msps *2 - fview.layers['m6a'] * 0.5 - (fview.layers['seq'] == b'-')


fv.tools.filter_regions(fview, base_name="msp", length_limits=(20, np.inf))



# -----------------------------------------------------------------------------
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



