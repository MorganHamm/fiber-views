#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  1 11:53:41 2025

@author: morgan
"""

import numpy as np
import pandas as pd

import os
from pathlib import Path
# import seaborn as sns
import matplotlib.pyplot as plt

import fiber_views as fv
# import anndata as ad

# run for pop-out plots in IPython interactive mode:
# %matplotlib qt

# an example Fiber-seq BAM file. This file is of the MM2d arabidopsis cell line,
# aligned to TAIR10 and only covers the first 100k bases of chr3
bam_path = fv.example_bam_path

# a bed file with gene annotations in the region covered by the example BAM
example_genes_bed = fv.example_bed_path

# the directory we will save the plots we make
figures_dir = Path(bam_path).parent.parent.absolute() / 'example_figures'



bed_data = fv.read_bed(example_genes_bed)
anno_df = fv.bed_to_anno_df(bed_data)


WINDOW = (-2000, 2000)


fview = fv.build_single_fview(bam_file=bam_path, site_info=anno_df.iloc[16, :], 
                mod_defs=fv.PB_FS_mod_defs, region_defs=fv.NUC_region_defs, 
                window=WINDOW, fully_span=False)

# The rest of the example codes will use the fview object created here...


# -----------------------------------------------------------------------------
# Plot of MSPs surrounding TSS site of a sinlge gene:

wd = 0.7
fig, ax = plt.subplots(figsize=(12, 6))
ax = fv.plot.make_plot_ax(fview, ax)
fv.plot.draw_fiber_bars(fview, ax, width=wd)
fv.plot.draw_regions(fview, ax, color="purple", width=wd)

ax.set_ylim(59.5, -1)

ax.set_title(f'MSPs around TSS of {fview.obs.gene_id.iloc[0]}, first 60 fibers, {fview.obs.site_name.iloc[0]}')
ax.set_xlabel('position from TSS')
ax.set_ylabel('fiber')
plt.tight_layout()

plt.savefig(figures_dir / "single_site_msp_plot.png")
plt.savefig(figures_dir / "single_site_msp_plot.svg")


# -----------------------------------------------------------------------------
# Adding CpG methylation state to the previous plot:
    
# we will use a function that adds a sparse matrix to our fiber-view that marks 
# the location of every CG di-nucleotide. 

# then when plotting, we will first color all CpG sites as Blue, then color over
# the methylated ones in red.

# the lines added from the previous exampel are marked with '# <---'

fv.tools.mark_cpg_sites(fview) # <---

wd = 0.7
fig, ax = plt.subplots(figsize=(12, 6))
ax = fv.plot.make_plot_ax(fview, ax)
fv.plot.draw_fiber_bars(fview, ax, width=wd)
fv.plot.draw_regions(fview, ax, color="purple", width=wd)
fv.plot.draw_mods(fview, ax, mod='cpg_sites', width=wd, color='blue') # <---
fv.plot.draw_mods(fview, ax, mod='cpg', width=wd, color='red') # <---

ax.set_ylim(59.5, -1)

ax.set_title(f'MSPs around TSS of {fview.obs.gene_id.iloc[0]}, first 60 fibers, {fview.obs.site_name.iloc[0]}')
ax.set_xlabel('position from TSS')
ax.set_ylabel('fiber')
plt.tight_layout()

plt.savefig(figures_dir / "single_site_msp_cpg_plot.png")
plt.savefig(figures_dir / "single_site_msp_cpg_plot.svg")


# -----------------------------------------------------------------------------
# aggregation
import seaborn as sns


# make a new region called fp (footpritn) of unmethylated patches < 100bp
fview = fv.tools.filter_regions(fview, base_name='nuc', new_base_name='fp', length_limits=(0,80))


fv.tools.mark_cpg_sites(fview)
fview_agg = fv.tools.agg_by_obs_and_bin(fview, obs_group_var='site_name', bin_width=30, 
                                        obs_to_keep=['seqid', 'pos', 'strand', 
                                                     'gene_id', 'score'])


plot_df = pd.DataFrame({
    'pos' : fview_agg.var['pos'],
    'm6a_rate' : (fview_agg.layers['m6a_count']/(fview_agg.layers['A_count'] + fview_agg.layers['T_count'])).flatten(),
    'cpg_rate' : (fview_agg.layers['cpg_count'] / fview_agg.layers['cpg_sites_count']).flatten(),
    'cpg_sites' : np.squeeze(fview_agg.layers['cpg_sites_count'] / fview_agg.layers['read_coverage']),
    'mcpg_rate' : np.squeeze(fview_agg.layers['cpg_count'] / fview_agg.layers['read_coverage']),
    })

fig, ax = plt.subplots(nrows=3, height_ratios=(3,.5,.5), figsize=(12, 8), sharex=True)


# msp_plot
ax[0] = fv.plot.make_plot_ax(fview, ax[0])
fv.plot.draw_fiber_bars(fview, ax[0], width=wd)
fv.plot.draw_regions(fview, ax[0], color="purple", width=wd)
fv.plot.draw_regions(fview, ax[0], base_name='fp', color="#00FF00", width=wd)
fv.plot.draw_mods(fview, ax[0], mod='cpg_sites', width=wd, color='blue') # <---
fv.plot.draw_mods(fview, ax[0], mod='cpg', width=wd, color='red') # <---
ax[0].set_ylim(49.5, -1)
ax[0].set_ylabel('fibers')
   
sns.lineplot(plot_df, x='pos', y='m6a_rate', color='purple', ax=ax[1])
ax[1].set_ylabel("% m6A")
# sns.lineplot(plot_df, x='pos', y='cpg_rate', color='red', ax=ax[2])
ax[2].fill_between(data=plot_df, x='pos', y1='cpg_sites', color='#B0B0B0')
ax[2].fill_between(data=plot_df, x='pos', y1='mcpg_rate', color='orange')
ax[2].set_xlabel('position from TSS')

ax[2].set_ylabel('CpG\nmethylation')
ax[2].set_ylabel('CpG abundance\n& methylation')
ax[2].set_xlabel('position from TSS')
ax[0].set_title(f'{fview.obs.gene_id.iloc[0]}, {fview.obs.site_name.iloc[0]}')

plt.tight_layout()

plt.savefig(figures_dir / "single_site_multiplot_fp.png")
plt.savefig(figures_dir / "single_site_multiplot_fp.svg")

