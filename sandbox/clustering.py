#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 17:00:52 2022

@author: morgan
"""

import numpy as np
import pandas as pd

import os
import time
import fiber_views as fv
import pysam

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

import seaborn as sns



from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.cluster import hierarchy

from scipy.spatial import distance


# -----------------------------------------------------------------------------


os.chdir(os.path.expanduser("~/git/fiber_views"))


# -----------------------------------------------------------------------------
# basic reading from bam file and summarizing 


bamfile = pysam.AlignmentFile("local/aligned.fiberseq.chr3_trunc.bam", "rb")

bed_data = fv.read_bed('local/TAIR10_genes.bed')
anno_df = fv.bed_to_anno_df(bed_data)
anno_df.query('seqid == "chr3" & pos < 200000', inplace=True) 

all_genes = fv.FiberView(bamfile, anno_df, window=(-2000, 2000), fully_span=True)

# make a list of all gene ids in the all_genes object
gene_ids = pd.unique(all_genes.obs.gene_id)

# create a subset object of 1 particular gene
temp_view = all_genes[all_genes.obs.gene_id == gene_ids[21]]

# region plot of temp_view
fv.tools.simple_region_plot(temp_view)


# check kmer clustering 
fv.tools.count_kmers(temp_view, k=6)
fv.tools.calc_kmer_dist(temp_view, metric='cityblock')
Z = linkage(distance.squareform(temp_view.obsp['kmer_dist']))
dend = dendrogram(Z)
temp_view = temp_view[hierarchy.leaves_list(Z), :]
temp_view.obs['kmer_cluster'] = dend['leaves_color_list']

# bin
temp_binned = fv.tools.agg_by_obs_and_bin(temp_view, obs_group_var=None, bin_width=20,
                                          obs_to_keep=temp_view.obs.columns)

# define the features to cluster by
fv_filt = temp_view.copy() # make a copy so regions are not removed from original
fv.tools.filter_regions(fv_filt, 'msp', (80, np.inf))
clust_feats = fv.tools.make_dense_regions(fv_filt, 'msp')


# run pca
pca = PCA(n_components=None, whiten=False)
pca.fit(clust_feats)
transf = pca.transform(clust_feats)
temp_view.obsm['PCA'] = transf

# plot of variance explained by each PC
sns.scatterplot(x=np.arange(pca.components_.shape[0]), y=pca.explained_variance_ratio_)

# copy some PCs to obs for plotting
temp_view.obs['PC1'] = temp_view.obsm['PCA'][:,0]
temp_view.obs['PC2'] = temp_view.obsm['PCA'][:,1]
temp_view.obs['PC3'] = temp_view.obsm['PCA'][:,2]

# k-means clustering
kmeans = KMeans(n_clusters=4, random_state=0).fit(transf[:,0:5])
temp_view.obs['km_clust'] = kmeans.labels_
sns.scatterplot(data=temp_view.obs, x='PC1', y='PC2', hue='km_clust', palette="tab10")

# re-order and plot
temp_view3 = temp_view[temp_view.obs.sort_values(by='km_clust').index]
fv.tools.simple_region_plot(temp_view3)


# aggregate each cluster, and plot a heatmap of MSP coverage
temp_view3.obs['clust'] = temp_view3.obs['km_clust'].astype(str)
clust_agg = fv.tools.agg_by_obs_and_bin(temp_view3, obs_group_var='clust', bin_width=10, 
                            obs_to_keep=['seqid', 'pos', 'strand', 'gene_id', 'score'])
sns.heatmap(clust_agg.layers['msp_coverage'] / clust_agg.layers['read_coverage'])


