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

def get_mod_pos_from_rec(rec, mods=M6A_MODS):
    # from Mitchell's extract_bed_from_bam.py
    # rec should be a pysam.libcalignedsegment.AlignedSegment (PileupRead.alignment)
    if rec.modified_bases_forward is None:
        return None
    positions = []
    for mod in mods:
        if mod in rec.modified_bases_forward:
            pos = np.array(rec.modified_bases_forward[mod], dtype=D_TYPE)[:, 0]
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
    alignment = read.alignment
    for i, mod_pos in enumerate(mod_positions):
        if alignment.is_reverse and use_strand:
            # mod positions depen on is_reverse bec they were called before alignment
            seq = Seq(alignment.query_sequence).reverse_complement()
        else:
            seq = Seq(alignment.query_sequence)
        print(str(i)+ ":  " + seq[mod_pos-offset:mod_pos] + ">" + 
              seq[mod_pos:mod_pos+offset+1])
    
    


# =============================================================================
# MAIN/TEST CODE
# =============================================================================

bamfile = pysam.AlignmentFile(
    "/home/morgan/git/pysam_and_anndata/aligned.fiberseq.chr3_trunc.bam", "rb")


reads = get_reads_at_center_pos(bamfile, "chr3:2005001")

print_aligned_reads(reads, offset=10)

x = 20
mod_type = CPG_MODS # M6A_MODS, CPG_MODS
read_x_mods = get_mod_pos_from_rec(reads[x].alignment, mods=M6A_MODS)
if reads[x].alignment.is_reverse:
    if mod_type == CPG_MODS:
        # CpGs the reverse base is offset by 1 as compared to A/Ts
        read_x_mods_2 = reads[x].alignment.query_length - read_x_mods - 2 # for cpg
    else:
        read_x_mods_2 = reads[x].alignment.query_length - read_x_mods - 1 # for m6A
else:
    read_x_mods_2
print_mod_contexts(reads[x], read_x_mods)
print_mod_contexts(reads[x], read_x_mods_2, use_strand=False)

