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


fv2 = fv.ad2fv(fview[fview.obs.gene_id == "AT3G01480", ])

fv2 = fv.ad2fv(fview[fview.obs.gene_id == "AT3G01150", ]) # this one has 2 clusters




fv.tools.count_kmers(fv2, k=6)

fv.tools.calc_kmer_dist(fv2, metric='cityblock')


# cluster sequences by kmer counts
Z = linkage(distance.squareform(fv2.obsp['kmer_dist']))
dendrogram(Z)



# re-order fiber view rows to match dendrogram
fv2 = fv.ad2fv(fv2[hierarchy.leaves_list(Z), :] )
fv.tools.plot_methylation(fv2)


# save sequences as fasta file for MSA
fv2.obs['unique_id']  = [str(idx) + ":" + read_name for (idx, read_name) in \
                         zip(fv2.obs.index, fv2.obs['read_name'])]
seqs = fv2.get_seq_records(id_col='unique_id')
SeqIO.write(seqs, 'local/chr3:51583.fa', 'fasta')

