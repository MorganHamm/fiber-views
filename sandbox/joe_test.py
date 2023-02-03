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


os.chdir('/net/gs/vol1/home/joemin/rotation_projects/W23_fiber_het/analysis_code/fiber_views')


# -----------------------------------------------------------------------------
# basic reading from bam file and summarizing


bamfile = pysam.AlignmentFile('local/aligned.fiberseq.bam', 'rb')

anno_df = pd.read_csv('/net/gs/vol1/home/joemin/extractions/ATCOPIA78/ATCOPIA78LTR_msa.csv')

# bed_data = fv.read_bed('local/TAIR10_genes.bed')
# anno_df = fv.bed_to_anno_df(bed_data)
# anno_df.query('seqid == "chr3" & pos < 200000', inplace=True)

all_genes = fv.FiberView(bamfile, anno_df, window=(-2000, 2000), fully_span=True)

# make a list of all gene ids in the all_genes object
gene_ids = pd.unique(all_genes.obs.gene_id)

# create a subset object of 1 particular gene
temp_view = all_genes[all_genes.obs.gene_id == gene_ids[69]]

# region plot of temp_view
plot = fv.tools.simple_region_plot(temp_view)
fig = plot.get_figure()
fig.savefig('test.png')
