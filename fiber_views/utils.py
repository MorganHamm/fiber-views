#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 16:05:34 2022

@author: morgan
"""
import numpy as np
import pandas as pd
import anndata as ad
from scipy.sparse import csr_matrix, coo_matrix
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

class GenomicPosition:
    def __init__(self, coord_str):
        [self.seqid, pos] = coord_str.split(":")
        self.pos = int(pos)
        

# =============================================================================
# FUNCTIONS
# =============================================================================

def get_mod_pos_from_rec(rec, mods=M6A_MODS, score_cutoff=200):
    # from Mitchell's extract_bed_from_bam.py
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


def get_reads_at_center_pos(alignment_file, genomic_coordinate_str):
    # genomic coordinate should be in form "chr3:200000"
    # returns list of pysam.libcalignedsegment.PileupRead objects
    ref_pos = GenomicPosition(genomic_coordinate_str)
    
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
    
    
def get_strand_correct_mods(read, mod_type=M6A_MODS, centered=False):
    # get modification positions and correct them to match the forward genomic strand
    raw_mods = get_mod_pos_from_rec(read.alignment, mods=mod_type)
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
  
def build_seq_array(reads, window_offset):
    # create a byte array of the sequences. 
    # warning, filter reads first
    char_array = np.empty((len(reads), 2*window_offset), dtype="S1")
    for i, read in enumerate(reads):
        center_pos = read.query_position
        seq = read.alignment.query_sequence[center_pos - window_offset : center_pos + window_offset]
        char_array[i, :] = np.frombuffer(seq.encode('UTF-8'), dtype="S1")
    return(char_array)


def build_mod_array(reads, window_offset, mod_type=M6A_MODS, sparse=True):
    I = []
    J = []
    none_count = 0
    for i, read in enumerate(reads):
        mods = get_strand_correct_mods(read, mod_type, centered=True)
        # print(i)
        if mods is None:
            none_count += 1
            continue
        mods = mods + window_offset
        mods = [mod for mod in mods if mod >= 0 and mod <= 2*window_offset-1]
        for mod in mods:
            I.append(i)
            J.append(mod)
    V = np.ones((len(J)), dtype=bool)
    mod_mtx = coo_matrix((V, (I, J)), shape=(len(reads), 2*window_offset))
    print(none_count)
            
    return(mod_mtx)



# =============================================================================
# MAIN/TEST CODE
# =============================================================================


import os
os.chdir(os.path.expanduser("~/git/fiber_views"))

bamfile = pysam.AlignmentFile("local/aligned.fiberseq.chr3_trunc.bam", "rb")


reads = get_reads_at_center_pos(bamfile, "chr3:2015001")

print_aligned_reads(reads, offset=50)



x = 50
mod_type = M6A_MODS
mods = get_strand_correct_mods(reads[x], mod_type)

print_mod_contexts(reads[x], mods, use_strand=False)
print(reads[x].alignment.is_reverse)

win_offset = 2000
filtered_reads = filter_reads_by_window(reads, win_offset)


seq_array = build_seq_array(filtered_reads, win_offset)

cpg_array = build_mod_array(filtered_reads, win_offset, mod_type=CPG_MODS)
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




