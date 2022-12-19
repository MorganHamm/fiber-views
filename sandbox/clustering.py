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
from sklearn.metrics import silhouette_score, adjusted_rand_score

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
temp_view = all_genes[all_genes.obs.gene_id == gene_ids[69]]

# region plot of temp_view
fv.tools.simple_region_plot(temp_view)


# check kmer clustering 
fv.tools.count_kmers(temp_view, k=6)
fv.tools.calc_kmer_dist(temp_view, metric='cityblock')
Z = linkage(distance.squareform(temp_view.obsp['kmer_dist']))
dend = dendrogram(Z)
temp_view = temp_view[hierarchy.leaves_list(Z), :]
temp_view.obs['kmer_cluster'] = dend['leaves_color_list']
# temp_view = temp_view[temp_view.obs.kmer_cluster == "C1"]


# define the features to cluster by
fv_filt = fv.tools.filter_regions(temp_view, 'msp', (100, np.inf))
temp_binned = fv.tools.agg_by_obs_and_bin(fv_filt, obs_group_var=None, bin_width=20,
                                          obs_to_keep=temp_view.obs.columns)
# clust_feats = fv.tools.make_dense_regions(fv_filt, 'msp')
# clust_feats = fv.tools.make_dense_regions(temp_view, 'msp')
clust_feats = temp_binned.layers['msp_coverage']
clust_feats = np.concatenate([temp_binned.layers['m6a_count']], axis=1)

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
kmeans = KMeans(n_clusters=8, random_state=None, n_init=20).fit(temp_view.obsm['PCA'][:,0:40])
temp_view.obs['km_clust'] = kmeans.labels_
sns.scatterplot(data=temp_view.obs, x='PC1', y='PC3', hue='km_clust', palette="tab10")
# sns.scatterplot(data=temp_view.obs, x='PC3', y='PC2', hue='kmer_cluster', palette="tab10")

# re-order and plot
temp_view3 = temp_view[temp_view.obs.sort_values(by='km_clust').index]
temp_view3 = fv_filt[temp_view.obs.sort_values(by='km_clust').index] # show the filtered view
fv.tools.simple_region_plot(temp_view3)


# aggregate each cluster, and plot a heatmap of MSP coverage
temp_view3 = temp_view[temp_view.obs.sort_values(by='km_clust').index]
temp_view3.obs['clust'] = temp_view3.obs['km_clust'].astype(str)
clust_agg = fv.tools.agg_by_obs_and_bin(temp_view3, obs_group_var='clust', bin_width=10, 
                            obs_to_keep=['seqid', 'pos', 'strand', 'gene_id', 'score'])
sns.heatmap(clust_agg.layers['msp_coverage'] / clust_agg.layers['read_coverage'])


# plot the component loadings of the first 10 PCs
sns.heatmap(pca.components_[0:10, ], cmap="PiYG", center=0)


# -----------------------------------------------------------------------------
# explore different values of k and # of PCAs


row_list = []

for k in np.arange(2, 16):
    for n_PCs in np.arange(2, 8):
        kmeans_list = []
        score_list = []
        ARI_list = []
        for seed in np.arange(40):
            kmeans = KMeans(n_clusters=k, random_state=(seed + 200), n_init=1).fit(temp_view.obsm['PCA'][:,0:n_PCs])
            kmeans_list.append(kmeans.labels_)
            sil_score = silhouette_score(X=temp_view.obsm['PCA'][:,0:n_PCs],
                                         labels = kmeans.labels_)
            score_list.append(sil_score)
            if seed > 0:
                ARI_list.append(adjusted_rand_score(kmeans_list[seed], kmeans_list[seed - 1]))
        new_row = pd.Series({'k':k, 'n_PCs':n_PCs, 
                             'score_mean':np.mean(score_list),
                             'score_max':np.max(score_list),
                             'score_sdev':np.std(score_list),
                             'ARI_min':np.min(ARI_list),
                             'ARI_mean':np.mean(ARI_list),
                             'ARI_max':np.max(ARI_list),
                             'ARI_sdev':np.std(ARI_list)})
        row_list.append(new_row)

results = pd.DataFrame(row_list)



g = sns.PairGrid(data=results, y_vars=['score_mean', 'ARI_min', 'ARI_mean'], x_vars=['k'], hue='n_PCs', palette="tab10")
g.map(sns.lineplot)
g.add_legend()



g = sns.PairGrid(data=results, y_vars=['score_max', 'ARI_min', 'ARI_mean'], x_vars=['k'], hue='n_PCs')
g.map(sns.lineplot)
g.add_legend()
