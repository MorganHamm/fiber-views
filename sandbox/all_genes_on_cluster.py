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

bamfile = pysam.AlignmentFile(os.path.expanduser(
    "~/fiber_seq/nobackup/fiberseq-smk/results/PS00137/aligned_TAIR10/aligned.fiberseq.bam"), 
    "rb")

bed_data = fv.read_bed('local/TAIR10_genes.bed')
bed_data.query('not chrom in ["chrC", "chrM"]', inplace=True)
anno_df = fv.bed_to_anno_df(bed_data)


print("collecting records for {} genes".format(anno_df.shape[0]))

start_time = time.time()
fview = fv.FiberView(bamfile, anno_df)
end_time = time.time()

print("time to process all genes: {} minutes".format( (end_time - start_time)/60  ))

start_time = time.time()
sdata = fview.summarize_by_obs(cols_to_keep=list(anno_df.keys()))
end_time = time.time()

print("time to summarize: {} minutes".format( (end_time - start_time)/60  ))

fview.write_h5ad("local/all_genes.h5ad")

sdata.write_h5ad("local/all_genes_summary.h5ad")
