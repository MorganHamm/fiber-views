#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  4 11:11:35 2022

@author: morgan
"""

import os
import fiber_views as fv

from scipy.spatial import distance
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.cluster import hierarchy

from Bio import SeqIO

os.chdir(os.path.expanduser("~/git/fiber_views"))


fview = fv.read_h5ad("local/test_fiberview.h5ad")
# fview = fv.FiberView(bamfile, anno_df, window=(-2000, 2000), fully_span=True)


fv2 = fv.ad2fv(fview[fview.obs.gene_id == "AT3G01480", ])

fv2 = fv.ad2fv(fview[fview.obs.gene_id == "AT3G01150", ]) # this one has 2 clusters


fv2 = fv.ad2fv(fview[fview.obs.gene_id == "AT3G01270", ])


# count kmers and calcuate distances
fv.tools.count_kmers(fv2, k=6)
fv.tools.calc_kmer_dist(fv2, metric='cityblock')


# cluster sequences by kmer counts
Z = linkage(distance.squareform(fv2.obsp['kmer_dist']))
dend = dendrogram(Z)



# re-order fiber view rows to match dendrogram
fv2 = fv.ad2fv(fv2[hierarchy.leaves_list(Z), :] )
fv2.obs['kmer_cluster'] = dend['leaves_color_list']

# sort by read quality
# fv2 = fv2[fv2.obs.sort_values('rq').index]

fv2 = fv2[fv2.obs.kmer_cluster == 'C2']

fv.tools.simple_region_plot(fv2, mod='cpg')

sns.scatterplot(x=range(fv2.shape[0]), y=fv2.obs.rq)

fv.tools.plot_methylation(fv2)


# save sequences as fasta file for MSA
fv2.obs['unique_id']  = [str(idx) + ":" + read_name for (idx, read_name) in \
                         zip(fv2.obs.index, fv2.obs['read_name'])]
seqs = fv2.get_seq_records(id_col='unique_id')
SeqIO.write(seqs, 'local/chr3:51583.fa', 'fasta')

fv2.obs['m6a_counts'] = np.sum(fv2.layers['m6a'].toarray(), axis=1)
fv2.obs['cpg_counts'] = np.sum(fv2.layers['cpg'].toarray(), axis=1)
fv2.obs['msp_cov'] = np.sum(fv.tools.make_dense_regions(fv2, 'msp'), axis=1)


sns.scatterplot(x=range(fv2.shape[0]), y=fv2.obs.cpg_cov)

sns.scatterplot(x=np.log10(1.00001-fv2.obs.rq), y=fv2.obs.msp_cov)
sns.scatterplot(x=np.arange(fv2.shape[0]), y=fv2.obs.cpg_counts)


np.corrcoef(x=np.arange(fv2.shape[0]), y=fv2.obs.msp_cov) 

np.corrcoef(x=np.log(1.00001-fv2.obs.rq), y=fv2.obs.msp_cov)
