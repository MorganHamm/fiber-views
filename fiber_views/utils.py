#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 16:05:34 2022

@author: morgan
"""

# TODO: make strand specific.
# TODO: bed to df function

import numpy as np
import pandas as pd
import anndata as ad
from scipy.sparse import csr_matrix, coo_matrix, vstack
print(ad.__version__)
import pysam

from Bio.Seq import Seq
import re


CPG_MODS = [("C", 0, "m")]
M6A_MODS = [("A", 0, "a"), ("T", 1, "a")]
D_TYPE = np.int64



# =============================================================================
# CLASSES
# =============================================================================

# class GenomicPosition:
#     def __init__(self, coord_str):
#         [self.seqid, pos] = coord_str.split(":")
#         self.pos = int(pos)

class GenomicPosition:
    def __init__(self):
        self.seqid = None
        self.pos = None
        self.strand = None
    def from_str(self, coord_str):
        [self.seqid, pos] = coord_str.split(":")
        self.pos = int(pos)
        self.strand = "*"
        return(self)
    def from_series(self, series):
        # like from a pandas dataframe row
        self.seqid = series.loc['seqid']
        self.pos = series.loc['pos']
        self.strand = series.loc['strand']
        return(self)
    def __repr__(self):
        return("A genomic position object\nseqid: {}, pos: {}, strand: {}"
               .format(self.seqid, self.pos, self.strand))
    def __str__(self):
        return(self.__repr__())
    
    
# =============================================================================
# FUNCTIONS
# =============================================================================

def get_mod_pos_from_rec(rec, mods=M6A_MODS, score_cutoff=200):
    # from Mitchell's extract_bed_from_bam.py (modified)
    # rec should be a pysam.libcalignedsegment.AlignedSegment (PileupRead.alignment)
    if rec.modified_bases_forward is None:
        return None
    positions = []
    for mod in mods:
        if mod in rec.modified_bases_forward:
            mod_score_array = np.array(rec.modified_bases_forward[mod], dtype=D_TYPE)
            pos = mod_score_array[mod_score_array[:,1] >= score_cutoff, 0]
            # pos = np.array(rec.modified_bases_forward[mod], dtype=D_TYPE)[:, 0]
            positions.append(pos)
    if len(positions) < 1:
        return None
    mod_positions = np.concatenate(positions, dtype=D_TYPE)
    mod_positions.sort(kind="mergesort")
    return mod_positions


def get_reads_at_center_pos(alignment_file, genomic_coordinate):
    # genomic coordinate should be in form "chr3:200000"
    # returns list of pysam.libcalignedsegment.PileupRead objects
    ref_pos = genomic_coordinate
    
    pileup_iter = alignment_file.pileup(ref_pos.seqid, ref_pos.pos, ref_pos.pos +1 )  
    
    for pileup_column in pileup_iter:
        if pileup_column.pos == ref_pos.pos:
            reads = [read for read in pileup_column.pileups 
                     if not read.query_position is None]
            return(reads)
        elif pileup_column.pos > ref_pos.pos:
            return(None)
    return(None)

def print_aligned_reads(reads, offset=5):
    # test function to make sure reads are aligning correctly
    for i, read in enumerate(reads):
        center_pos = read.query_position
        alignment = read.alignment
        if alignment.is_reverse:
            # alignment.query_sequence is on forward genomic strand
            seq = Seq(alignment.query_sequence)# .reverse_complement()
        else:
            seq = Seq(alignment.query_sequence)
        print(str(i)+ ":  " + seq[center_pos-offset:center_pos] + ">" + 
              seq[center_pos:center_pos+offset+1])
        

def print_mod_contexts(read, mod_positions, offset=5, use_strand=True):
    # test function make sure mods are aligned correctly
    alignment = read.alignment
    for i, mod_pos in enumerate(mod_positions):
        if alignment.is_reverse and use_strand:
            # mod positions depen on is_reverse bec they were called before alignment
            seq = Seq(alignment.query_sequence).reverse_complement()
        else:
            seq = Seq(alignment.query_sequence)
        print(str(i)+ ":  " + seq[mod_pos-offset:mod_pos] + ">" + 
              seq[mod_pos:mod_pos+offset+1])
    
    
def get_strand_correct_mods(read, mod_type=M6A_MODS, centered=False, score_cutoff=200):
    # get modification positions and correct them to match the forward genomic strand
    raw_mods = get_mod_pos_from_rec(read.alignment, mods=mod_type, score_cutoff=score_cutoff)
    if raw_mods is None:
        return(None)
    if read.alignment.is_reverse:
        if mod_type == CPG_MODS:
            # CpGs the reverse base is offset by 1 as compared to A/Ts
            mods = read.alignment.query_length - raw_mods - 2 # for cpg
        else:
            mods = read.alignment.query_length - raw_mods - 1 # for m6A
        mods = np.flip(mods)
    else:
        mods = raw_mods
    if centered:
        mods = mods - read.query_position
    return(mods)
  
def filter_reads_by_window(reads, window_offset):
    # filter reads to only those that fill the window around the center position
    return([read for read in reads if read.query_position - window_offset >= 0 and 
     read.query_position + window_offset <= read.alignment.query_length])
  
def build_seq_array(reads, window_offset, strand="+"):
    # create a byte array of the sequences. 
    # warning, filter reads first
    char_array = np.empty((len(reads), 2*window_offset), dtype="S1")
    for i, read in enumerate(reads):
        center_pos = read.query_position
        seq = read.alignment.query_sequence[center_pos - window_offset : center_pos + window_offset]
        if strand == "-":
            seq = str(Seq(seq).reverse_complement())
        char_array[i, :] = np.frombuffer(seq.encode('UTF-8'), dtype="S1")
    return(char_array)


def build_mod_array(reads, window_offset, mod_type=M6A_MODS, strand="+", sparse=True, score_cutoff=200):
    I = []
    J = []
    none_count = 0
    for i, read in enumerate(reads):
        mods = get_strand_correct_mods(read, mod_type, centered=True, score_cutoff=score_cutoff)
        if mods is None:
            none_count += 1
            continue
        mods = mods + window_offset
        if strand == "-":
            mods = (2*window_offset-1) - mods - 1 * (mod_type == CPG_MODS)
        mods = [mod for mod in mods if mod >= 0 and mod <= 2*window_offset-1]
        for mod in mods:
            I.append(i)
            J.append(mod)
    V = np.ones((len(J)), dtype=bool)
    mod_mtx = coo_matrix((V, (I, J)), shape=(len(reads), 2*window_offset))
    if sparse == False:
        mod_mtx = mod_mtx.toarray()       
    return(mod_mtx)

def build_row_anno_from_reads(reads, anno_series):
    row_data_dict = anno_series.to_dict()
    row_data_dict['read_name'] = [read.alignment.query_name for read in reads]
    row_data_dict['read_length'] = [read.alignment.query_length for read in reads]
    row_data_dict['read_flag'] = [read.alignment.flag for read in reads]
    return(pd.DataFrame(row_data_dict))


def build_anndata_from_df(alignment_file, df, window_offset=1000):
    row_anno_df_list = []
    seq_mtx_list = []
    m6a_mtx_list = []
    cpg_mtx_list = []
    for i, row in df.iterrows():
        reads = get_reads_at_center_pos(alignment_file, 
                                        GenomicPosition().from_series(row))
        reads = filter_reads_by_window(reads, window_offset)
        row_anno_df_list.append(build_row_anno_from_reads(reads, row))
        seq_mtx_list.append(build_seq_array(reads, window_offset, 
                                            strand=row.loc['strand']))
        m6a_mtx_list.append(build_mod_array(reads, window_offset, 
                                            mod_type=M6A_MODS, sparse=True, 
                                            score_cutoff=220,
                                            strand=row.loc['strand']))
        cpg_mtx_list.append(build_mod_array(reads, window_offset, 
                                            mod_type=CPG_MODS, sparse=True, 
                                            score_cutoff=220,
                                            strand=row.loc['strand']))
    adata = ad.AnnData(vstack(m6a_mtx_list).tocsr())
    adata.layers['seq'] = np.vstack(seq_mtx_list)
    adata.layers['m6a'] = vstack(m6a_mtx_list).tocsr()
    adata.layers['cpg'] = vstack(cpg_mtx_list).tocsr()
    adata.obs = pd.concat(row_anno_df_list)
    adata.obs_names = ["{}:{}({}) {}".format(obs_row.seqid, obs_row.pos, 
                                             obs_row.strand, obs_row.read_name) 
                       for i, obs_row in adata.obs.iterrows()]
    adata.var_names = np.arange(-window_offset, window_offset)
    adata.var = pd.DataFrame({"pos" : adata.var_names})
    return(adata)

# =============================================================================
# MAIN/TEST CODE
# =============================================================================


import os
os.chdir(os.path.expanduser("~/git/fiber_views"))

bamfile = pysam.AlignmentFile("local/aligned.fiberseq.chr3_trunc.bam", "rb")

anno_df = pd.DataFrame({
    "seqid" : ["chr3", "chr3", "chr3"],
    "pos" : [2015001, 3000000, 3000000],
    "strand" : ["+", "-", "+"],
    "gene_id" : ["gene_1", "gene_2", "gene_3"],
    "score" : [56, 600, 40]
    })


fview = build_anndata_from_df(bamfile, anno_df)




# check that cpgs are landing on Cs...
fv2 = fview[fview.obs.gene_id == "gene_2"]

seq_array = fv2.layers['seq'].copy()

temp = seq_array[fv2.layers['cpg'].toarray()]

for i in range(100):
    print(temp[i])



# -----------------------------------------------------------------------------


reads = get_reads_at_center_pos(bamfile, GenomicPosition().from_str("chr3:2015001"))

print_aligned_reads(reads, offset=50)



# x = 50
# mod_type = M6A_MODS
# mods = get_strand_correct_mods(reads[x], mod_type)
# print_mod_contexts(reads[x], mods, use_strand=False)
# print(reads[x].alignment.is_reverse)

win_offset = 2000
filtered_reads = filter_reads_by_window(reads, win_offset)


seq_array = build_seq_array(filtered_reads, win_offset)

cpg_array = build_mod_array(filtered_reads, win_offset, mod_type=CPG_MODS, sparse=False)
m6a_array = build_mod_array(filtered_reads, win_offset, mod_type=M6A_MODS)

bytes(seq_array[0,:]).decode('UTF-8')[350]



for i in np.arange(3000,3030):
    I = m6a_array.row[i]
    J = m6a_array.col[i]
    seq = bytes(seq_array[I,:]).decode('UTF-8')
    center_pos = J
    offset = 10
    print(str(i)+ ":  " + seq[center_pos-offset:center_pos] + ">" + 
          seq[center_pos:center_pos+offset+1])

# -----------------------------------------------------------------------------

read = reads[5]
temp = np.array(read.alignment.modified_bases_forward[CPG_MODS[0]])

anno_df = pd.DataFrame({
    "seqid" : ["chr3", "chr3", "chr3"],
    "pos" : [2015001, 3000000, 4000000],
    "strand" : ["+", "-", "-"],
    "gene_id" : ["gene_1", "gene_2", "gene_3"],
    "score" : [56, 600, 40]
    })

for i, row in anno_df.iterrows():
    print(row)
    row.loc['seqid']
    type(row) == pd.core.series.Series

row_df_dict = row.to_dict()
vals = np.arange(9)
row_df_dict['values'] = vals
temp = build_row_anno_from_reads(reads, row)

seq_array