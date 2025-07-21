"""Main module."""




import numpy as np
import pandas as pd
import anndata as ad
from scipy.sparse import csr_matrix, coo_matrix, vstack
import pysam
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from . import utils


# deffinitions for base mods --------------------------------------------------

# PacBio Fiber-seq
PB_FS_mod_defs = [
    {'name' : 'm6a', 'mod_code' : [("A", 0, "a"), ("T", 1, "a")], 'threshold' : 125, 'rev_offset' : 0},
    {'name' : 'cpg', 'mod_code' : [("C", 0, "m")], 'threshold' : 125, 'rev_offset' : 1}
    ]

# ONT Fiber-seq
ONT_FS_mod_defs = [
    {'name' : 'm6A', 'mod_code' : [("A", 0, "a"), ("T", 1, "a")], 'threshold' : 220, 'rev_offset' : 0},
    {'name' : '5mCpG', 'mod_code' : [("C", 0, "m")], 'threshold' : 220, 'rev_offset' : 1},
    {'name' : '5hmC', 'mod_code' : [("C", 0, "h"), ("G", 1, "h")], 'threshold' : 220, 'rev_offset' : 0}
    ]

# deffinitions for regions ----------------------------------------------------

# BAM with nucleosomes called
NUC_region_defs = [
    {'name' : 'nuc', 'tags' : ('ns', 'nl')},
    {'name' : 'msp', 'tags' : ('as', 'al')}
    ]

# BAM with MSPs with FIRE scores
FIRE_region_defs = [
    {'name' : 'nuc', 'tags' : ('ns', 'nl')},
    {'name' : 'msp', 'tags' : ('as', 'al', 'aq')}
    ]

# No regions called
NONE_regions_def = []

# -----------------------------------------------------------------------------


def build_single_fview(bam_file, site_info, mod_defs, region_defs, window=(-1000, 1000), 
            fully_span=True, region_interval=30, filter_args={'dist':3000, 'cutoff':2},
            tags=['np', 'ec', 'rq'], max_reads=300):
    # site_info can be dict or pd.Series with keys 'seqid', 'pos', 'strand'
    
    bamfile = pysam.AlignmentFile(bam_file, "rb")
    
    reads = utils.ReadList().get_reads(bamfile, 
                                 ref_pos=(site_info['seqid'], site_info['pos'], site_info['strand']),
                                 max_reads=max_reads)
    if fully_span:
        reads.filter_by_window(window, inplace=True)
    if filter_args is not None:
        reads.filter_by_end_meth(dist=filter_args['dist'], cutoff=filter_args['cutoff'], inplace=True)
    if len(reads) == 0:
        return(None)
    
    obs = reads.build_anno_df(site_info, tags=tags)
    
    # initialize anndata
    fview = ad.AnnData(
        obs=obs, var=pd.DataFrame({"pos" : np.arange(window[0], window[1])}))
    
    # seq layer
    fview.layers['seq'] = reads.build_seq_array(window)
    
    # mod layers
    fview.uns['mods'] = []
    for mod_def in mod_defs:
        fview.layers[mod_def['name']] = reads.build_mod_array_from_def(window, 
                                                    mod_def=mod_def, sparse=True)
        fview.uns['mods'].append(mod_def['name'])
        
    # region layers
    fview.uns['region_base_names'] = []
    for region_def in region_defs:
            reg_pos_mtx, reg_len_mtx, reg_score_mtx = \
                reads.build_sparse_region_array(window, tags=region_def['tags'], 
                                                interval=region_interval)
            base_name = str(region_def['name'])
            fview.layers[base_name + '_pos'] = reg_pos_mtx.tocsr()
            fview.layers[base_name + '_len'] = reg_len_mtx.tocsr()
            fview.layers[base_name + '_score'] = reg_score_mtx.tocsr()
            fview.uns['region_base_names'].append(region_def['name'])
    
    
    # UNS
    fview.uns['region_report_interval'] = region_interval
    fview.uns['mod_defs'] = mod_defs
    fview.uns['region_defs'] = region_defs
    fview.uns['bin_width'] = 1
    
    fview.X = csr_matrix(fview.shape) # empty matrix, needed for AnnData.to_memory()
    
    return(fview)


def build_multi_fview(bam_file, sites_df, mod_defs, region_defs, window=(-1000, 1000), 
            fully_span=True, region_interval=30, filter_args={'dist':3000, 'cutoff':2},
            tags=['np', 'ec', 'rq'], max_reads=300):
    
    fview_list = []
    for i, site_info in sites_df.iterrows():
        fview_chunk = build_single_fview(bam_file=bam_file, site_info=site_info, 
                    mod_defs=mod_defs, region_defs=region_defs, window=window, 
                    fully_span=fully_span, region_interval=region_interval, 
                    filter_args=filter_args, tags=tags, max_reads=max_reads)
        fview_list.append(fview_chunk)
        
    fview = ad.concat(fview_list, axis=0, merge='same', uns_merge='first', index_unique="_")
    fview.obs.reset_index(inplace=True, drop=True)
    
    return(fview)



