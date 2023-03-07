#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 15:24:10 2023

@author: morgan
"""
import numpy as np
import pandas as pd

import os
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


region = pd.DataFrame({'seqid':['chr3'], 'pos':[30000], 'strand':['+']})


fview = fv.FiberView(bamfile, region, window=(-10000,10000), fully_span=False)

fv.tools.simple_region_plot(fview)

fv.tools.filter_regions(fview, 'msp', new_base_name='lnk', 
                        length_limits=(0, 150), inplace=True)
fv.tools.filter_regions(fview, 'msp', length_limits=(151, np.inf), inplace=True)

fview_agg = fv.tools.agg_by_obs_and_bin(fview, obs_group_var='site_name', bin_width=20)

hmap_data = pd.DataFrame(np.concatenate([fview_agg.layers['lnk_coverage']/fview_agg.layers['read_coverage'], 
                                         fview_agg.layers['msp_coverage']/fview_agg.layers['read_coverage'],
                                         fview_agg.layers['cpg_count']/fview_agg.layers['cpg_site_count']], 
                                        axis=0), columns=fview_agg.var['pos'])

fig, axs = plt.subplots(nrows=2, sharex=False)
plt.subplot(211)
fv.tools.simple_region_plot(fview, mod='cpg')
plt.subplot(212)
sns.heatmap(hmap_data, cmap="YlGnBu", mask=hmap_data.isnull())

sns.heatmap(fview_agg.layers['lnk_coverage']/fview_agg.layers['read_coverage'])
