#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  4 11:11:35 2022

@author: morgan
"""


import numpy as np
import pandas as pd

import os
import sys
import seaborn as sns
import matplotlib.pyplot as plt
import fiber_views as fv
import pysam

import itertools

from scipy.spatial import distance

from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.cluster import hierarchy

from Bio.Align import MultipleSeqAlignment
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO

os.chdir(os.path.expanduser("~/git/fiber_views"))


fview = fv.read_h5ad("local/test_fiberview.h5ad")
fv2 = fv.ad2fv(fview[fview.obs.gene_id == "AT3G01480", ])

fv2 = fv.ad2fv(fview[fview.obs.gene_id == "AT3G01150", ]) # this one has 2 clusters

# fv2 = fview




def count_kmers(fiber_view, k):
    # count kmers in each fiber
    kmer_to_idx = {bytes(k) : v for (k, v) in \
                   zip(itertools.product(b'ACGT', repeat=k), range(4**k))}
    fiber_view.obsm['kmers'] = np.zeros((fv2.shape[0], 4**k))
    for m in range(fiber_view.shape[1] - k + 1):
        kmer_col = fiber_view.layers['seq'][:, m:m+k]
        for i, row in enumerate(kmer_col):
            fiber_view.obsm['kmers'][i, kmer_to_idx[bytes(row)]] += 1
    fiber_view.uns['kmer_len'] = k
    # kmer_idx is the column index of the kmer matrix
    fiber_view.uns['kmer_idx'] = list(kmer_to_idx.keys())
    return(None)


def calc_kmer_dist(fiber_view, metric='euclidean'):
    dists =  distance.pdist(fiber_view.obsm['kmers'], metric=metric)
    fiber_view.obsp['kmer_dist'] = distance.squareform(dists)
    return(None)

def get_seq_records(fiber_view, id_col="read_name"):
    seqs = [Seq(bytes(row)) for row in fv2.layers['seq']]
    seq_records = []
    for i in range(fiber_view.shape[0]):
        seq_records.append( SeqRecord(seqs[i], id=fiber_view.obs[id_col][i], 
                                      description=fiber_view.obs.index[i]) )
    return(seq_records)


count_kmers(fv2, k=6)

calc_kmer_dist(fv2, metric='cityblock')
# calc_kmer_dist(fv2, metric='euclidean')


Z = linkage(distance.squareform(fv2.obsp['kmer_dist']))

dendrogram(Z)

fv2 = fv2[hierarchy.leaves_list(Z), :] # re-order rows to match dendrogram

fv2.obs['unique_id']  = [str(idx) + ":" + read_name for (idx, read_name) in \
                         zip(fv2.obs.index, fv2.obs['read_name'])]

seqs = get_seq_records(fv2, id_col='unique_id')

SeqIO.write(seqs, 'local/chr3:51583.fa', 'fasta')
# -----------------------------------------------------------------------------
# junk 




k = 3 # kmer length
kmer_to_idx = {bytes(k) : v for (k, v) in zip(itertools.product(b'ACGT', repeat=k), 
                                          range(4**k))}
fv2.obsm['kmers'] = np.zeros((fv2.shape[0], 4**k))
for m in range(fv2.shape[1] - k + 1):
    kmer_col = fv2.layers['seq'][:, m:m+k]
    # kmers = [bytes(row) for row in kmer_col]
    
    for i, row in enumerate(kmer_col):
        fv2.obsm['kmers'][i, kmer_to_idx[bytes(row)]] += 1
    





itertools.combinations_with_replacement([b'A', b'C', b'G', b'T'], 3)

vals = [b'A', b'C', b'G', b'T']

key_tuples = list(itertools.product(vals, repeat=3))

[bytes(row) for row in list(itertools.product(*[vals] * 3))]




[bytes(row) for row in itertools.product(b'ACGT', repeat=3)]

temp = zip(itertools.product(b'ACGT', repeat=3), range(4**3))

mapping = {bytes(k) : v for (k, v) in temp}

k = 3
mapping = {bytes(k) : v for (k, v) in zip(itertools.product(b'ACGT', repeat=k), 
                                          range(4**k))}



kmer_col = fv2.layers['seq'][:, 0:6]

kmers = [bytes(row) for row in kmer_col]