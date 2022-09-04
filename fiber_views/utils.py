#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 16:05:34 2022

@author: morgan
"""


import numpy as np
import pandas as pd
import anndata as ad
from scipy.sparse import csr_matrix, coo_matrix, vstack
import pysam
from Bio.Seq import Seq



CPG_MODS = [("C", 0, "m")]
M6A_MODS = [("A", 0, "a"), ("T", 1, "a")]
D_TYPE = np.int64


# =============================================================================
# CLASSES
# =============================================================================

class ReadList(list):
    """
    A simple list of pysam.libcalignedsegment.PileupRead objects, plus methods
    useful for constructing anndata elements from the read objects. also tracks
    strand info of the genomic query position
    """
    def __init__(self, normal_list=[], strand="+"):
        super().__init__()
        self.strand = strand
        self.extend(normal_list)
    def get_reads(self, alignment_file, ref_pos):
        """Example: reads=ReadList().get_reads(bamfile, ('chr3', 200000, '+'))
        """
        pileup_iter = alignment_file.pileup(ref_pos[0], ref_pos[1], ref_pos[1] +1 )   
        for pileup_column in pileup_iter:
            if pileup_column.pos == ref_pos[1]:
                reads = [read for read in pileup_column.pileups 
                         if not read.query_position is None]
                break
            elif pileup_column.pos > ref_pos[1]:
                break
        self.clear()
        self.strand = ref_pos[2]
        self.extend(reads)
        return(self)        
    def filter_by_window(self, window_offset, inplace=False):
        """remove reads that don't fully span a window of +/- window_offset.
            if inplace == True: this instance is modified, else: a new ReadList 
            object is returned"""
        list_out = [read for read in self if read.query_position - window_offset >= 0 and 
                read.query_position + window_offset <= read.alignment.query_length]
        if inplace:
            self.clear()
            self.extend(list_out)
            return(None)
        else:
            return(ReadList(list_out, self.strand))
    def build_seq_array(self, window_offset, strand=None):
        # create a byte array of the sequences. 
        # warning, filter reads first
        if strand is None:
            strand = self.strand
        char_array = np.empty((len(self), 2*window_offset), dtype="S1")
        for i, read in enumerate(self):
            center_pos = read.query_position
            seq = read.alignment.query_sequence[center_pos - window_offset : 
                                                center_pos + window_offset]
            if strand == "-":
                seq = str(Seq(seq).reverse_complement())
            char_array[i, :] = np.frombuffer(seq.encode('UTF-8'), dtype="S1")
        return(char_array)
    def build_mod_array(self, window_offset, mod_type=M6A_MODS, strand=None, 
                        sparse=True, score_cutoff=200):
        if strand is None:
            strand = self.strand
        I = []
        J = []
        none_count = 0
        for i, read in enumerate(self):
            mods = get_strand_correct_mods(read, mod_type, centered=True, 
                                           score_cutoff=score_cutoff)
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
        mod_mtx = coo_matrix((V, (I, J)), shape=(len(self), 2*window_offset))
        if sparse == False:
            mod_mtx = mod_mtx.toarray()       
        return(mod_mtx)
    def build_anno_df(self, anno_series):
        row_data_dict = anno_series.to_dict()
        row_data_dict['read_name'] = [read.alignment.query_name for read in self]
        row_data_dict['read_length'] = [read.alignment.query_length for read in self]
        row_data_dict['read_flag'] = [read.alignment.flag for read in self]
        row_data_dict['site_name'] = "{}:{}({})".format(anno_series.seqid, 
                                                        anno_series.pos, 
                                                        anno_series.strand)
        return(pd.DataFrame(row_data_dict))
    def print_aligned_centers(self, offset=5):
        # test function to make sure reads are aligning correctly
        for i, read in enumerate(self):
            center_pos = read.query_position
            alignment = read.alignment
            seq = Seq(alignment.query_sequence)
            print(str(i)+ ":  " + seq[center_pos-offset:center_pos] + ">" + 
                  seq[center_pos:center_pos+offset+1])
        



    
    
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


def read_bed(bed_file): 
    # bed file should follow bed standard and not include column names
    bed_data = pd.read_csv(bed_file, sep="\t", header=None)
    BED_HEADER = ['chrom', 'start', 'end', 'name', 'score', 'strand', 'thick_start',
                  'thick_end', 'item_rgb', 'block_count', 'block_widths', 'block_starts']
    bed_data.set_axis(BED_HEADER[0:bed_data.shape[1]], axis=1, inplace=True)
    return(bed_data)

def bed_to_anno_df(bed_df, entry_name_type="gene_id"):
    anno_df = pd.DataFrame({
        "seqid" : bed_df.chrom,
        "pos" : bed_df.start * (bed_df.strand == "+")  + 
        bed_df.end * (bed_df.strand == "-"),
        "strand" : bed_df.strand,
        entry_name_type : bed_df.name,
        "score" : bed_df.score,
        })
    return(anno_df)



    
    
    
    