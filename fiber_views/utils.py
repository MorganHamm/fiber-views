#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 16:05:34 2022

@author: morgan
"""


import numpy as np
import pandas as pd
from itertools import repeat
import warnings

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
    strand info of the genomic query position.
    
    Parameters
    ----------
    normal_list : list, optional
        A list of pysam.libcalignedsegment.PileupRead objects. The default is [].
    strand : str, optional
        The strand of the genomic query position. The default is "+".
    
    Attributes
    ----------
    strand : str
        The strand of the genomic query position.
    """

    def __init__(self, normal_list=[], strand="+"):
        super().__init__()
        self.strand = strand
        self.extend(normal_list)

    def get_reads(self, alignment_file, ref_pos, max_reads):
        """
        Retrieve reads from a BAM file for a given reference position.
        
        Parameters
        ----------
        alignment_file : pysam.AlignmentFile
            A BAM file opened with pysam.
        ref_pos : tuple
            A tuple containing the reference name, position, and strand of the
            genomic query position. The tuple should be of the form
            (reference_name, position, strand).
        max_reads : int 
            the max number of reads to load from the bam file, usefull to speed
            up processing when coverage is deep.
            
        Returns
        -------
        self : ReadList
            The `ReadList` object with the reads and strand information.
            
        Example
        -------
        reads = ReadList().get_reads(bamfile, ('chr3', 200000, '+'))
        """
        pileup_iter = alignment_file.pileup(
            ref_pos[0], ref_pos[1], ref_pos[1] + 1)
        reads = []
        for pileup_column in pileup_iter:
            if pileup_column.pos == ref_pos[1]:
                for read in pileup_column.pileups:
                    if not read.query_position is None:
                        reads.append(read)
                    if len(reads) >= max_reads:
                        break
                break
            elif pileup_column.pos > ref_pos[1]:
                break
        self.clear()
        self.strand = ref_pos[2]
        self.extend(reads)
        return(self)

    def filter_by_window(self, window, inplace=False, strand=None):
        """
        Remove reads that do not fully span a given window of +/- window_offset.
        
        Parameters
        ----------
        window : tuple
            A tuple of integers representing the window of +/- window_offset. The
            tuple should be of the form (window_start, window_end).
        inplace : bool, optional
            If True, the `ReadList` object is modified in place. If False, a new
            `ReadList` object is returned. The default is False.
        strand : str, optional
            The strand of the genomic query position. If not provided, the strand
            information of the `ReadList` object is used. The default is None.
            
        Returns
        -------
        None or ReadList
            If `inplace` is True, returns None. If `inplace` is False, returns
            a new `ReadList` object with the filtered reads and strand information.
        """
        window_len = window[1] - window[0]
        if strand is None:
            strand = self.strand
        if strand == "-":
            window = (-window[1], -window[0])
        list_out = [read for read in self if read.query_position + window[0] >= 0 and
                    read.query_position + window[1] <= read.alignment.query_length]
        if inplace:
            self.clear()
            self.extend(list_out)
            return(None)
        else:
            return(ReadList(list_out, self.strand))

    def filter_by_end_meth(self, dist=3000, cutoff=2, inplace=False):
        """
        Remove reads if they have fewer than [cutoff] m6A mods within [dist] base 
        pairs of the read start and end.
        
        Parameters
        ----------
        dist : int, optional
            The distance in base pairs from the read start and end to consider.
            The default is 3000.
        cutoff : int, optional
            The minimum number of m6A mods within [dist] base pairs of the read 
            start and end required to keep the read. The default is 2.
        inplace : bool, optional
            If True, the `ReadList` object is modified in place. If False, a new
            `ReadList` object is returned. The default is False.
            
        Returns
        -------
        None or ReadList
            If `inplace` is True, returns None. If `inplace` is False, returns
            a new `ReadList` object with the filtered reads and strand information.
        """

        list_out = []
        for read in self:
            mods = get_strand_correct_mods(read)
            read_len = read.alignment.query_length
            if mods is None:
                continue
            if sum(mods < dist) > cutoff and sum(mods > (read_len - dist)) > cutoff:
                list_out.append(read)
        if inplace:
            self.clear()
            self.extend(list_out)
            return(None)
        else:
            return(ReadList(list_out, self.strand))                
        

    def build_seq_array(self, window, strand=None):
        """
        Create a byte array of the sequences for the reads in the `ReadList` object.
        
        Parameters
        ----------
        window : tuple
            A tuple of integers representing the window of +/- window_offset. The
            tuple should be of the form (window_start, window_end).
        strand : str, optional
            The strand of the genomic query position. If not provided, the strand
            information of the `ReadList` object is used. The default is None.
            
        Returns
        -------
        char_array : numpy array
            A byte array of the sequences for the reads in the `ReadList` object.
            
        Notes
        -----
        Make sure to filter the reads in the `ReadList` object before using this 
        method.
        """
        window_len = window[1] - window[0]
        if strand is None:
            strand = self.strand
        if strand == "-":
            window = (-window[1], -window[0])            
        char_array = np.empty((len(self), window_len), dtype="S1")
        for i, read in enumerate(self):
            center_pos = read.query_position

            L_pad = max(0 - (center_pos + window[0]), 0)
            R_pad = max((center_pos + window[1]) - len(read.alignment.query_sequence), 0)
            seq = read.alignment.query_sequence[max(center_pos + window[0], 0):
                                                min(center_pos + window[1], len(read.alignment.query_sequence))]
            if strand == "-":
                seq = "-" * R_pad + str(Seq(seq).reverse_complement()) + "-" * L_pad
            else:
                seq = "-" * L_pad +  str(seq) + "-" * R_pad
            char_array[i, :] = np.frombuffer(seq.encode('UTF-8'), dtype="S1")
            
        return(char_array)

    def build_mod_array(self, window, mod_type=M6A_MODS, strand=None,
                        sparse=True, score_cutoff=200):
        """
        Create a base modification matrix for the reads in the `ReadList` object.
        
        Parameters
        ----------
        window : tuple
            A tuple of integers representing the window of +/- window_offset. The
            tuple should be of the form (window_start, window_end).
        mod_type : list, optional
            A list of tuples representing the base modification type to consider.
            The default is M6A_MODS.
        strand : str, optional
            The strand of the genomic query position. If not provided, the strand
            information of the `ReadList` object is used. The default is None.
        sparse : bool, optional
            If True, the base odification matrix is returned in sparse format. If 
            False, the matrix is returned in dense format. The default is True.
        score_cutoff : int, optional
            The minimum score required for a base modification to be considered. The
            default is 200.
            
        Returns
        -------
        mod_mtx : numpy array or scipy sparse matrix
            A base modification matrix for the reads in the `ReadList` object.
        """
        window_len = window[1] - window[0]
        if strand is None:
            strand = self.strand
        I = []
        J = []
        for i, read in enumerate(self):
            mods = get_strand_correct_mods(read, mod_type, centered=True,
                                           score_cutoff=score_cutoff)
            if mods is None:
                continue
            if strand == "-":
                # mods = (window_len-1) - mods - 1 * (mod_type == CPG_MODS)
                mods = -mods - 1 - 1 * (mod_type == CPG_MODS)
            mods = mods - window[0]                
            mods = [mod for mod in mods if mod >= 0 and mod <= window_len-1]
            for mod in mods:
                I.append(i)
                J.append(mod)
        V = np.ones((len(J)), dtype=bool)
        mod_mtx = coo_matrix((V, (I, J)), shape=(len(self), window_len))
        if sparse == False:
            mod_mtx = mod_mtx.toarray()
        return(mod_mtx)


    def build_sparse_region_array(self, window, tags=('ns', 'nl'),
                                  interval=30, strand=None):
        """
        Create a sparse region matrix for the reads in the `ReadList` object.
        
        Parameters
        ----------
        window : tuple
            A tuple of integers representing the window of +/- window_offset. The
            tuple should be of the form (window_start, window_end).
        tags : tuple, optional
            A tuple of tags to consider. The default is ('ns', 'nl').
        interval : int, optional
            The interval at which to report region information. This determines
            the minimum window size that can be subset to that still preserve
            region info. The default is 30.
        strand : str, optional
            The strand of the genomic query position. If not provided, the strand
            information of the `ReadList` object is used. The default is None.
            
        Returns
        -------
        region_mtx : scipy sparse matrix
            A sparse region matrix for the reads in the `ReadList` object.
        """
        window_len = window[1] - window[0]
        if strand is None:
            strand = self.strand
        region_df = pd.DataFrame()
        for i, read in enumerate(self):
            starts, lengths, scores = get_strand_correct_regions(
                read, tags=tags, centered=True)
            if strand == "-":
                starts = np.flip(0 - np.array(starts) - np.array(lengths))
                lengths = np.flip(lengths)
                scores = np.flip(scores)
            starts = np.array(starts, dtype=int) - window[0]
            region_chunk = pd.DataFrame({
                'row' : i,
                'start' : starts,
                'length' : lengths,
                'score' : scores
                })
            region_df = pd.concat([region_df, region_chunk])
        region_df = region_df[(region_df.start < window_len) &
                              (region_df.start + region_df.length > 0) ]
        return(make_sparse_regions(region_df, 
                                   shape=(len(self), window_len), 
                                   interval=interval))

    def build_anno_df(self, anno_series, tags=['np', 'ec', 'rq']):
        """
        Create a data frame with annotation data for the reads in the `ReadList` object.
        
        Parameters
        ----------
        anno_series : pandas Series
            A pandas Series containing annotation data for the genomic query position.
        tags : list, optional
            A list of tags to include in the data frame. The default is ['np', 'ec', 'rq'].
            
        Returns
        -------
        df : pandas DataFrame
            A data frame with annotation data for the reads in the `ReadList` object.
        """
        row_data_dict = anno_series.to_dict()
        row_data_dict['read_name'] = [
            read.alignment.query_name for read in self]
        row_data_dict['read_length'] = [
            read.alignment.query_length for read in self]
        row_data_dict['read_flag'] = [read.alignment.flag for read in self]
        for tag in tags:
            row_data_dict[tag] = [read.alignment.get_tag(tag) for read in self]
        row_data_dict['site_name'] = "{}:{}({})".format(anno_series.seqid,
                                                        anno_series.pos,
                                                        anno_series.strand)
        return(pd.DataFrame(row_data_dict))

    def print_aligned_centers(self, offset=5):
        """
        Print the center positions of the reads in the `ReadList` object with a specified number of bases on either side.
        This is a test function to make sure reads are aligning correctly
        
        Parameters
        ----------
        offset : int, optional
            The number of bases on either side of the center position to include in the output. The default is 5.
            
        Returns
        -------
        None
        """
        for i, read in enumerate(self):
            center_pos = read.query_position
            alignment = read.alignment
            seq = Seq(alignment.query_sequence)
            print(str(i) + ":  " + seq[center_pos-offset:center_pos] + ">" +
                  seq[center_pos:center_pos+offset+1])


# =============================================================================
# FUNCTIONS
# =============================================================================

def get_mod_pos_from_rec(rec, mods=M6A_MODS, score_cutoff=200):
    """
    Retrieve positions of modified bases in a record.
    
    Parameters
    ----------
    rec : pysam.libcalignedsegment.AlignedSegment
        A record containing modified bases.
    mods : list, optional
        A list of modified bases to consider, in the form (base, index, code).
        The default is M6A_MODS.
    score_cutoff : int, optional
        The minimum score required for a modified base to be included.
        The default is 200.
        
    Returns
    -------
    mod_positions : numpy.ndarray
        An array of positions of modified bases.
        
    Example
    -------
    mod_positions = get_mod_pos_from_rec(read.alignment)
    """
    # from Mitchell's extract_bed_from_bam.py (modified)
    # rec should be a pysam.libcalignedsegment.AlignedSegment (PileupRead.alignment)
    if rec.modified_bases_forward is None:
        return None
    positions = []
    for mod in mods:
        if mod in rec.modified_bases_forward:
            mod_score_array = np.array(
                rec.modified_bases_forward[mod], dtype=D_TYPE)
            pos = mod_score_array[mod_score_array[:, 1] >= score_cutoff, 0]
            # pos = np.array(rec.modified_bases_forward[mod], dtype=D_TYPE)[:, 0]
            positions.append(pos)
    if len(positions) < 1:
        return None
    mod_positions = np.concatenate(positions, dtype=D_TYPE)
    mod_positions.sort(kind="mergesort")
    return mod_positions


def print_mod_contexts(read, mod_positions, offset=5, use_strand=True):
    """
    Print the contexts surrounding modified bases in a read.
    
    Parameters
    ----------
    read : pysam.libcalignedsegment.AlignedSegment
        A read containing modified bases.
    mod_positions : numpy.ndarray
        An array of positions of modified bases in the read.
    offset : int, optional
        The number of bases on either side of the modified base to include in
        the context. The default is 5.
    use_strand : bool, optional
        Whether to use the strand information in the read to determine the
        context. If True, the context will be reversed if the read is on the
        negative strand. The default is True.
        
    Example
    -------
    print_mod_contexts(read, mod_positions)
    """
    # test function make sure mods are aligned correctly
    alignment = read.alignment
    for i, mod_pos in enumerate(mod_positions):
        if alignment.is_reverse and use_strand:
            # mod positions depen on is_reverse bec they were called before alignment
            seq = Seq(alignment.query_sequence).reverse_complement()
        else:
            seq = Seq(alignment.query_sequence)
        print(str(i) + ":  " + seq[mod_pos-offset:mod_pos] + ">" +
              seq[mod_pos:mod_pos+offset+1])


def get_strand_correct_mods(read, mod_type=M6A_MODS, centered=False, score_cutoff=200):
    """
    Retrieve modified bases in a read and correct their positions to match the forward genomic strand.
    
    Parameters
    ----------
    read : pysam.libcalignedsegment.AlignedSegment
        A read containing modified bases.
    mod_type : list, optional
        A list of modified bases to consider, in the form (base, index, code).
        The default is M6A_MODS.
    centered : bool, optional
        Whether to center the positions around the query position of the read.
        The default is False.
    score_cutoff : int, optional
        The minimum score required for a modified base to be included.
        The default is 200.
        
    Returns
    -------
    mods : numpy.ndarray
        An array of positions of modified bases, corrected for strand.
        
    Example
    -------
    mods = get_strand_correct_mods(read)
    """
    # get modification positions and correct them to match the forward genomic strand
    raw_mods = get_mod_pos_from_rec(
        read.alignment, mods=mod_type, score_cutoff=score_cutoff)
    if raw_mods is None:
        return(None)
    if read.alignment.is_reverse:
        if mod_type == CPG_MODS:
            # CpGs the reverse base is offset by 1 as compared to A/Ts
            mods = read.alignment.query_length - raw_mods - 2  # for cpg
        else:
            mods = read.alignment.query_length - raw_mods - 1  # for m6A
        mods = np.flip(mods)
    else:
        mods = raw_mods
    if centered:
        mods = mods - read.query_position
    return(mods)


def get_strand_correct_regions(read, tags=('ns', 'nl'), centered=False):
    """
    Retrieve start positions, lengths, and scores of regions in a read and correct them to match the forward genomic strand.
    
    Parameters
    ----------
    read : pysam.libcalignedsegment.AlignedSegment
        A read containing regions.
    tags : tuple, optional
        A tuple of tags containing the start positions, lengths, and scores of the
        regions. The default is ('ns', 'nl').
    centered : bool, optional
        Whether to center the positions around the query position of the read.
        The default is False.
        
    Returns
    -------
    starts : numpy.ndarray
        An array of start positions of regions, corrected for strand.
    lengths : numpy.ndarray
        An array of lengths of regions.
    scores : numpy.ndarray
        An array of scores of regions.
        
    Example
    -------
    starts, lengths, scores = get_strand_correct_regions(read)
    """
    for tag in tags:
        if not read.alignment.has_tag(tag):
            # return empty lists if any tag is missing
            return(([], [], []))
    raw_starts = np.array(read.alignment.get_tag(tags[0]), dtype='int32')
    raw_lengths = np.array(read.alignment.get_tag(tags[1]), dtype='int32')
    # raw_ends = raw_starts + raw_lengths
    if len(tags) == 3:
        raw_scores = np.array(read.alignment.get_tag(tags[2]), dtype='int32')
    else:
        raw_scores = list(repeat(1, len(raw_starts)))
    if read.alignment.is_reverse:
        starts = np.flip(read.alignment.query_length -
                         raw_starts - raw_lengths)
        # starts = np.flip(read.alignment.query_length -
        #                  raw_starts - raw_lengths)
        lengths = np.flip(raw_lengths)
        scores = np.flip(raw_scores)
    else:
        starts = raw_starts
        lengths = raw_lengths
        scores = raw_scores
    if centered:
        starts = starts - read.query_position
    return((starts, lengths, scores))


def read_bed(bed_file):
    """
    Read a BED file and return a pandas DataFrame.

    Parameters
    ----------
    bed_file : str
        The file path of the BED file to be read.
        The bed file should follow bed standard and not include column names.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the data from the BED file.
    """
    bed_data = pd.read_csv(bed_file, sep="\t", header=None)
    BED_HEADER = ['chrom', 'start', 'end', 'name', 'score', 'strand', 'thick_start',
                  'thick_end', 'item_rgb', 'block_count', 'block_widths', 'block_starts']
    bed_data.set_axis(BED_HEADER[0:bed_data.shape[1]], axis=1, inplace=True)
    return(bed_data)


def bed_to_anno_df(bed_df, entry_name_type="gene_id"):
    """
    Convert a data frame in BED format to another data frame with a different layout.
    
    Parameters
    ----------
    bed_df : pandas.DataFrame
        Data frame in BED format, with columns 'chrom', 'start', 'end', 'strand', 'name', and 'score'.
    entry_name_type : str, optional
        Column name for the unique identifier for each feature. The default is "gene_id".
    
    Returns
    -------
    anno_df : pandas.DataFrame
        Data frame with columns 'seqid', 'pos', 'strand', 'entry_name_type', and 'score'.
    """
    anno_df = pd.DataFrame({
        "seqid": bed_df.chrom,
        "pos": bed_df.start * (bed_df.strand == "+") +
        bed_df.end * (bed_df.strand == "-"),
        "strand": bed_df.strand,
        entry_name_type: bed_df.name,
        "score": bed_df.score,
    })
    return(anno_df)

   
def make_sparse_regions(region_df, shape, bin_width = 1, interval = 30):
    """Make a sparse matrix representing genomic regions.

    This function takes a DataFrame containing region information, as well as the
    shape of the resulting matrix and other parameters, and returns three sparse
    matrices representing the positions, lengths, and scores of the regions within
    the matrix.

    Parameters
    ----------
    region_df : pandas.DataFrame
        A DataFrame containing the region information. The DataFrame should have
        columns for `row`, `start`, `length`, and `score`, representing the row
        index of the matrix, the starting position of the region (0-based), the
        length of the region, and the score associated with the region, respectively.
    shape : tuple
        The shape of the resulting matrix before binning. The first element should be the number
        of rows in the matrix, and the second element should be the number of
        columns.
    bin_width : int, optional
        The width of each bin in the resulting matrix, in base pairs. The default
        is 1.
    interval : int, optional
        The interval at which to report region information. This determines the
        minimum window size that can be subset to that still preserve region info.
        The default is 30. interval is in number of bins, not bp

    Returns
    -------
    tuple of scipy.sparse.coo_matrix
        A tuple containing three sparse matrices representing the positions,
        lengths, and scores of the regions within the matrix. The matrices are in
        the form of COO sparse matrices. position values are still in base pairs 
        after binning. and may be negative for the first reported pos of a region
    """
    I = []  # row index in matrix
    J = []  # column index in matrix
    P = []  # position along region
    L = []  # total length of region
    S = []  # score associated with that region
    max_bin = shape[1] // bin_width
    new_shape = (shape[0], max_bin)
    old_row = -1
    for i, region in region_df.iterrows():
        if region.row != old_row:
            row_Js = [] # clear the list of taken J values for a new row
            old_row = region.row
        start_bin = min(max(region.start // bin_width, 0), max_bin)
        end_bin = min(max( (region.start + region.length) // bin_width, 0), max_bin)        
        J_vals = [start_bin] + list(
            np.arange(start_bin + (interval - (start_bin % interval)),
                      end_bin - 1,
                      interval)
        )
        # if the reporting positon is already taken, slide over until it's not
        J_new = []
        for k, val in enumerate(J_vals):
            while val in row_Js:
                val += 1
                if val - J_vals[k] >= interval:
                    warnings.warn("Warning: region report lost due to overloading")
                    break
            row_Js.append(val)
            J_new.append(val)
        I += [region.row] * len(J_new)
        J += J_new
        P += list(np.array(J_new) * bin_width - region.start)
        L += [region.length] * len(J_new)
        S += [region.score] * len(J_new)
    region_array_pos = coo_matrix((P, (I, J)), shape=new_shape, dtype=int)
    region_array_len = coo_matrix((L, (I, J)), shape=new_shape, dtype=int)
    region_array_score = coo_matrix((S, (I, J)), shape=new_shape, dtype=int)
    return(region_array_pos, region_array_len, region_array_score)
   
    