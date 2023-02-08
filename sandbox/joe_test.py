#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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

root_dir = '/net/gs/vol1/home/joemin/rotation_projects/W23_fiber_het/analysis_code/fiber_views'
os.chdir(root_dir)

# -----------------------------------------------------------------------------
bamfile = pysam.AlignmentFile('local/aligned.fiberseq.bam', 'rb')

output_dir = f'{root_dir}/figures'
family_name = 'ATCOPIA78'
endings = ['LTR']
ending = endings[0]
anno_df = pd.read_csv(f'/net/gs/vol1/home/joemin/te_extractions/{family_name}/{family_name}{ending}_msa.csv')
anno_df.query('pos < 10000', inplace=True)

all_positions = fv.FiberView(bamfile, anno_df, window=(-2000, 2000), fully_span=True)

# Unique positions correspond with the individual loci (e.g. the ~900 loci for ONSEN LTR)
pos_s = pd.unique(all_positions.obs.pos)
te_family_view = all_positions[all_positions.obs.pos == pos_s[0]]
plot = fv.tools.simple_region_plot(te_family_view)
fig = plot.get_figure()
fig.savefig(f'{output_dir}/{family_name}{ending}.svg') # For saving in different formats: https://www.marsja.se/how-to-save-a-seaborn-plot-as-a-file-e-g-png-pdf-eps-tiff/

# Dendritic tree of kmer similarity
# fv.tools.count_kmers(te_family_view, k=5)
# fv.tools.calc_kmer_dist(te_family_view, metric='cityblock')
# Z = linkage(distance.squareform(te_family_view.obsp['kmer_dist']))
# dend = dendrogram(Z)
# te_family_view = te_family_view[hierarchy.leaves_list(Z), :]
# te_family_view.obs['kmer_cluster'] = dend['leaves_color_list']

# Collapse down by site (i.e. "seqid:pos (strand)")
obs_group_var = 'site_name'
binned_view = fv.tools.agg_by_obs_and_bin(te_family_view, obs_group_var=obs_group_var, bin_width=20,
                                          obs_to_keep=all_positions.obs.columns)

cluster_features = binned_view.layers['msp_coverage']
cluster_features = np.concatenate([binned_view.layers['m6a_count']], axis=1)

pca = PCA(n_components=None, whiten=False)
pca.fit(cluster_features)
transformed = pca.transform(cluster_features)
binned_view.obsm['PCA'] = transformed

# plot of variance explained by each PC
plot = sns.scatterplot(x=np.arange(pca.components_.shape[0]), y=pca.explained_variance_ratio_)
fig = plot.get_figure()
fig.savefig(f'{output_dir}/{family_name}{ending}_PC_scatter.svg')

# copy some PCs to obs for plotting
binned_view.obs['PC1'] = binned_view.obsm['PCA'][:,0]
binned_view.obs['PC2'] = binned_view.obsm['PCA'][:,1]
binned_view.obs['PC3'] = binned_view.obsm['PCA'][:,2]

# k-means clustering
kmeans = KMeans(n_clusters=8, random_state=None, n_init=20).fit(binned_view.obsm['PCA'][:,0:40])
binned_view.obs['km_clust'] = kmeans.labels_
plot = sns.scatterplot(data=binned_view.obs, x='PC1', y='PC2', hue='km_clust', palette="tab10")
fig = plot.get_figure()
fig.savefig(f'{output_dir}/{family_name}{ending}_kmeans_scatter.svg')
# sns.scatterplot(data=temp_view.obs, x='PC3', y='PC2', hue='kmer_cluster', palette="tab10")

# re-order and plot
binned_view_sorted = binned_view[binned_view.obs.sort_values(by='km_clust').index]
binned_view_sorted = fv_filt[binned_view.obs.sort_values(by='km_clust').index] # show the filtered view
plot = fv.tools.simple_region_plot(binned_view_sorted)
fig = plot.get_figure()
fig.savefig(f'{output_dir}/{family_name}{ending}_sorted_by_cluster.svg')

# aggregate each cluster, and plot a heatmap of MSP coverage
binned_view_sorted = binned_view[binned_view.obs.sort_values(by='km_clust').index]
binned_view_sorted.obs['clust'] = binned_view_sorted.obs['km_clust'].astype(str)
clustered_aggregated_view = fv.tools.agg_by_obs_and_bin(binned_view_sorted, obs_group_var='clust', bin_width=10,
                            obs_to_keep=['seqid', 'pos', 'strand', 'gene_id', 'score'])
plot = sns.heatmap(clustered_aggregated_view.layers['msp_coverage'] / clustered_aggregated_view.layers['read_coverage'])
fig = plot.get_figure()
fig.savefig(f'{output_dir}/{family_name}{ending}_sorted_by_cluster_and_aggregated.svg')


# plot the component loadings of the first 10 PCs
plot = sns.heatmap(pca.components_[0:10, ], cmap="PiYG", center=0)
fig = plot.get_figure()
fig.savefig(f'{output_dir}/{family_name}{ending}_10_PCs.svg')
