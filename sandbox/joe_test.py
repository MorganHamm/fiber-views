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

all_positions = fv.FiberView(bamfile, anno_df, window=(-2000, 2000), fully_span=False)

# Unique positions correspond with the individual loci (e.g. the ~900 loci for ONSEN LTR)
pos_s = pd.unique(all_positions.obs.pos)
single_position_view = all_positions[all_positions.obs.pos == pos_s[0]]
plot = fv.tools.simple_region_plot(single_position_view)
fig = plot.get_figure()
fig.savefig(f'{output_dir}/{family_name}{ending}.svg') # For saving in different formats: https://www.marsja.se/how-to-save-a-seaborn-plot-as-a-file-e-g-png-pdf-eps-tiff/

# fv_filt = fv.tools.filter_regions(temp_view, 'msp', (100, np.inf))
binned_by_position_view = fv.tools.agg_by_obs_and_bin(single_position_view, obs_group_var=['seqid', 'pos'], bin_width=20,
                                          obs_to_keep=all_positions.obs.columns)
plot = fv.tools.simple_region_plot(binned_by_position_view)
fig = plot.get_figure()
fig.savefig(f'{output_dir}/{family_name}{ending}_binned.svg')
print(binned_by_position_view.head)

# clust_feats = fv.tools.make_dense_regions(fv_filt, 'msp')
# clust_feats = fv.tools.make_dense_regions(temp_view, 'msp')
# clust_feats = temp_binned.layers['msp_coverage']
