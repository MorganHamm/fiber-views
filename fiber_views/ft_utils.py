#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 16:05:34 2022

@author: morgan
"""


import numpy as np
import pandas as pd
# from itertools import repeat
import warnings

# import anndata as ad
from scipy.sparse import csr_matrix, coo_matrix, vstack
# import pysam
import pyft
from Bio.Seq import Seq



def get_reads(bam_file, anno_df):
    """

    Parameters
    ----------
    bam_file : str
        path to bam file
    anno_df : pd.DataFrame
        dataframe with at least these columns: 'seqid', 'pos', 'strand'

    Returns
    -------
    None.

    """
    fiberbam = pyft.Fiberbam(bam_file)
    chunk_list = []
    for i, site in anno_df.iterrows():
        fibers = fiberbam.fetch(site.seqid, site.pos, site.pos+1)
        fibers = [fiber for fiber in fibers]
        centers = [fiber.lift_query_positions([site.pos])[0] for fiber in fibers]
        lengths = [fiber.get_seq_length() for fiber in fibers]
        q_strands = [fiber.strand for fiber in fibers]
        read_ids = [fiber.qname for fiber in fibers]
        chunk = pd.DataFrame({'seqid': site.seqid,
                              'pos': site.pos,
                              'strand': site.strand,
                              'read_id': read_ids,
                              'center': centers,
                              'q_len': lengths,
                              'q_strand': q_strands,
                              'fiberdata': fibers
                              })
        chunk_list.append(chunk)
    fiber_df = pd.concat(chunk_list,ignore_index=True)
    return(fiber_df)
        

def build_seq_array(fiber_df, window):
    window_len = window[1] - window[0]
    char_array = np.empty((len(fiber_df), window_len), dtype="S1")
    i = 0
    for ind_name, row in fiber_df.iterrows():
        qw_strand = int(~np.logical_xor(row.strand == "+", row.q_strand == "+"))*2 -1
        s_window = np.array(window) * qw_strand
        s_window.sort()
        w_lims = row.center + s_window # window limits on the query
        
        L_pad = max(0 - w_lims[0], 0)
        R_pad = max(w_lims[1] - row.q_len, 0)
        
        seq = row.fiberdata.seq[max(w_lims[0], 0):min(w_lims[1], row.q_len)] 
                
        if qw_strand < 0:
            seq = "-" * R_pad + str(Seq(seq).reverse_complement()) + "-" * L_pad
        else:
            seq = "-" * L_pad +  str(seq) + "-" * R_pad
        char_array[i, :] = np.frombuffer(seq.encode('UTF-8'), dtype="S1")
  
        i += 1
    return(chra_array)
    
    
    
def build_mod_array(fiber_df, window, mod_type="m6a",  mapping='query', score_cutoff=200):
    # mod_type must be "m6a" or "cpg" to match the fiberdata attributes
    # mapping should be 'query' or 'reference'
    window_len = window[1] - window[0]
    I = []
    J = []
    V = []
    
    i = 0
    for ind_name, row in fiber_df.iterrows():

        
        mods = getattr(row.fiberdata, mod_type)
        scores = np.array(mods.ml)
        if mapping == "query":
            qw_strand = int(~np.logical_xor(row.strand == "+", row.q_strand == "+"))*2 -1
            s_window = np.array(window) * qw_strand
            s_window.sort()
            w_lims = row.center + s_window # window limits on the query
            
            starts = np.array(mods.starts)
        else:
            qw_strand = int(row.strand == "+")*2 - 1
            s_window = np.array(window) * qw_strand
            s_window.sort()
            w_lims = row.pos + s_window  # window limits on the reference
            starts = np.array(mods.reference_starts) 
            if row.q_strand == "-":
                scores = np.flip(scores)
            
        mod_arr = np.stack([starts, scores], axis=1)
        
        # filtering 
        mod_arr = mod_arr[np.isfinite(mod_arr[:,0])]
        mod_arr = mod_arr[ (mod_arr[:,0] > w_lims[0]) & (mod_arr[:,0] < w_lims[1]) & (mod_arr[:,1] > score_cutoff), : ]
        
        
        
        
        i += 1
    
# ---------------------


