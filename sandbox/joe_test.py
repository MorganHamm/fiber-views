#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
anno_df.query('pos < 20000', inplace=True)

all_positions = fv.FiberView(bamfile, anno_df, window=(-2000, 2000), fully_span=True)
# Layers with keys: seq, m6a, cpg, nuc_pos, nuc_len, nuc_score, msp_pos, msp_len, msp_score, cpg_sites

### Unique positions correspond with the individual loci (e.g. the ~900 loci for ONSEN LTR)
pos_s = pd.unique(all_positions.obs.pos)
te_family_view = all_positions
print(f'45: te_family_view.uns\n{te_family_view.layers}')
print(f'46: te_family_view.uns\n{te_family_view.uns}')
# print('Generating initial simple plot.')
# plot = fv.tools.simple_region_plot(te_family_view)
# fig = plot.get_figure()
# fig.savefig(f'{output_dir}/{family_name}{ending}.svg') # For saving in different formats: https://www.marsja.se/how-to-save-a-seaborn-plot-as-a-file-e-g-png-pdf-eps-tiff/
# fig.savefig(f'{output_dir}/{family_name}{ending}.png')
# plt.close(fig)


### Collapse down by site_name (i.e. "seqid:pos (strand)")
obs_group_var = 'site_name'
# Don't bin at first, just aggregate
agged_view = fv.tools.agg_by_obs_and_bin(te_family_view, obs_group_var=obs_group_var, bin_width=1,
                                          obs_to_keep=all_positions.obs.columns)
print(f'60: agged_view.layers\n{agged_view.layers}')
print(f'61: agged_view.uns\n{agged_view.uns}')

print('Generating binned plot.')
plot = sns.heatmap(agged_view.layers['nuc_coverage'])
fig = plot.get_figure()
fig.savefig(f'{output_dir}/{family_name}{ending}_binned.svg')
fig.savefig(f'{output_dir}/{family_name}{ending}_binned.png')
plt.close(fig)


### PCA
cluster_features = agged_view.layers['msp_coverage']
# cluster_features = np.concatenate([agged_view.layers['m6a_count']], axis=1)

pca = PCA(n_components=None, whiten=False)
pca.fit(cluster_features)

# print('Generating PC scatter plot.')
# plot = sns.scatterplot(x=np.arange(pca.components_.shape[0]), y=pca.explained_variance_ratio_)
# fig = plot.get_figure()
# fig.savefig(f'{output_dir}/{family_name}{ending}_PC_scatter.svg')
# fig.savefig(f'{output_dir}/{family_name}{ending}_PC_scatter.png')
# plt.close(fig)


### k-means clustering
print('Generating kmeans scatter plot.')
transformed = pca.transform(cluster_features)
agged_view.obsm['PCA'] = transformed
binned_and_filtered_view = fv.tools.filter_regions(agged_view, base_name='msp', length_limits=80)
agged_view.obs['PC1'] = agged_view.obsm['PCA'][:,0]
agged_view.obs['PC2'] = agged_view.obsm['PCA'][:,1]

kmeans = KMeans(n_clusters=8, random_state=None, n_init=20).fit(agged_view.obsm['PCA'][:,0:40])
agged_view.obs['km_clust'] = kmeans.labels_
plot = sns.scatterplot(data=agged_view.obs, x='PC1', y='PC2', hue='km_clust', palette="tab10")
fig = plot.get_figure()
fig.savefig(f'{output_dir}/{family_name}{ending}_kmeans_scatter.svg')
fig.savefig(f'{output_dir}/{family_name}{ending}_kmeans_scatter.png')
plt.close(fig)


### re-order and plot
print('Generating sorted cluster plot.')
agged_view_sorted = agged_view[agged_view.obs.sort_values(by='km_clust').index]
plot = sns.heatmap(agged_view_sorted.layers['nuc_coverage'])
fig = plot.get_figure()
fig.savefig(f'{output_dir}/{family_name}{ending}_sorted_by_cluster.svg')
fig.savefig(f'{output_dir}/{family_name}{ending}_sorted_by_cluster.png')
plt.close(fig)


### aggregate each cluster, and plot a heatmap of MSP coverage
print('Generating sorted aggregated cluster plot.')
agged_view_sorted = agged_view[agged_view.obs.sort_values(by='km_clust').index]
agged_view_sorted.obs['clust'] = agged_view_sorted.obs['km_clust'].astype(str)
clustered_aggregated_view = fv.tools.agg_by_obs_and_bin(agged_view_sorted, obs_group_var='clust', bin_width=10,
                            obs_to_keep=['seqid', 'pos', 'strand', 'gene_id', 'score'])
plot = sns.heatmap(clustered_aggregated_view.layers['msp_coverage'] / clustered_aggregated_view.layers['read_coverage'])
fig = plot.get_figure()
fig.savefig(f'{output_dir}/{family_name}{ending}_sorted_by_cluster_and_aggregated.svg')
fig.savefig(f'{output_dir}/{family_name}{ending}_sorted_by_cluster_and_aggregated.png')
plt.close(fig)


### plot the component loadings of the first 10 PCs
print('Generating 10 PCs plot.')
plot = sns.heatmap(pca.components_[0:10, ], cmap="PiYG", center=0)
fig = plot.get_figure()
fig.savefig(f'{output_dir}/{family_name}{ending}_10_PCs.svg')
fig.savefig(f'{output_dir}/{family_name}{ending}_10_PCs.png')
plt.close(fig)
