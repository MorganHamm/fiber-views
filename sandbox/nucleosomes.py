#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  9 15:59:59 2022

@author: morgan
"""

from fiber_views import utils

import numpy as np
import pandas as pd
import pysam
import os
from itertools import repeat
os.chdir(os.path.expanduser("~/git/fiber_views"))

bamfile = pysam.AlignmentFile("local/aligned.fiberseq.chr3_trunc.bam", "rb")

reads = utils.ReadList().get_reads(bamfile, ('chr3', 200000, '+'))

read = reads[0]

window_offset = 1000

strand = None

tags=('ns', 'nl')

def get_strand_correct_regions(read, tags=('ns', 'nl'), centered=False):
    # get starts, lengths and scores on the forward genomic strand for a 
    # single read
    for tag in tags:
        if not read.alignment.has_tag(tag):
            # return empty lists if any tag is missing
            return(([],[],[]))
    raw_starts = np.array(read.alignment.get_tag(tags[0]), dtype='int32')
    raw_lengths = np.array(read.alignment.get_tag(tags[1]))
    # raw_ends = raw_starts + raw_lengths
    if len(tags) == 3:
        raw_scores = np.array(read.alignment.get_tag(tags[2]), dtype='int8')
    else:
        raw_scores = list(repeat(1, len(raw_starts)))
    if read.alignment.is_reverse:
        starts = np.flip(read.alignment.query_length - raw_starts - raw_lengths)
        lengths = np.flip(raw_lengths)
        scores = np.flip(raw_scores)
    else:
        starts = raw_starts
        lengths = raw_lengths
        scores = raw_scores
    if centered:
        starts = starts - read.query_position
    return((starts, lengths, scores))


def build_region_array(reads, window_offset, tags=('ns', 'nl'), strand=None):
    # get 
    if strand is None:
        strand = reads.strand
    rows = []
    for i, read in enumerate(reads):
        starts, lengths, scores = get_strand_correct_regions(read, tags=tags, centered=True)
        starts = np.array(starts) + window_offset
        if strand == "-":
            starts = np.flip(2 * window_offset - starts)
            lengths = np.flip(lengths)
            scores = np.flip(scores)
        # filter for regions that overlap the window
        filtered_regions = []
        for region in zip(starts, lengths, scores):
            if region[0] < 0 :
                if region[0] + region[1] > 0:
                    # regions that overlap the start of the window
                    new_region = (0, region[1]+region[0], region[2])
                    filtered_regions.append(new_region)
            elif region[0] < 2*window_offset:
                if region[0] + region[1] > 2*window_offset:
                    # regions that overlap the end of the window
                    region = (region[0], 2*window_offset - region[0], region[2])
                filtered_regions.append(region)
        # construct a row of the matrix
        row = []
        last_end = 0
        for region in filtered_regions:
            row = row + list(repeat(0, region[0] - last_end)) + \
                list(repeat(region[2], region[1]))
            last_end = region[0] + region[1]
        row = row + list(repeat(0, 2*window_offset - last_end))
        rows.append(np.array(row, dtype='int8'))
    region_array = np.vstack(rows)
    return(region_array)

                    
            




        
