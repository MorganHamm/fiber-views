#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7 14:04:40 2022

@author: morgan
"""

import numpy as np
import pandas as pd

import os
import time
import fiber_views as fv
import pysam





os.chdir(os.path.expanduser("~/git/fiber_views"))

# =============================================================================
# bamfile = pysam.AlignmentFile(os.path.expanduser(
#     "~/fiber_seq/nobackup/fiberseq-smk/results/PS00137/aligned_TAIR10/aligned.fiberseq.bam"), 
#     "rb")
# 
# bed_data = fv.read_bed('local/TAIR10_genes.bed')
# bed_data.query('not chrom in ["chrC", "chrM"]', inplace=True)
# anno_df = fv.bed_to_anno_df(bed_data)
# 
# 
# print("collecting records for {} genes".format(anno_df.shape[0]))
# 
# start_time = time.time()
# fview = fv.FiberView(bamfile, anno_df)
# end_time = time.time()
# print("time to process all genes: {} minutes".format( (end_time - start_time)/60  ))
# 
# print("filtering MSPs")
# fv.tools.filter_regions(fview, base_name='msp', length_limits=(20, np.inf))
# fview.write_h5ad("local/all_genes.h5ad")
# =============================================================================
start_time = time.time()
print('loading data')
fview = fv.read_h5ad("local/all_genes.h5ad")
print("filtering out MSPs < 20bp")
fv.tools.filter_regions(fview, base_name='msp', length_limits=(20, np.inf))
end_time = time.time()
print("time to load and filter: {} minutes".format( (end_time - start_time)/60  ))

print("summarizing by site with 10bp windows")
start_time = time.time()
sdata = fv.tools.agg_by_obs_and_bin(fview, obs_group_var='site_name', bin_width=10, 
                                    obs_to_keep=['seqid', 'pos', 'strand', 
                                                 'gene_id', 'score'])
end_time = time.time()
print("time to summarize: {} minutes".format( (end_time - start_time)/60  ))


print("saving summary data")
sdata.write_h5ad("local/all_genes_summary.h5ad")
