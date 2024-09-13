"""Main module."""




import numpy as np
import pandas as pd
import anndata as ad
from scipy.sparse import csr_matrix, coo_matrix, vstack
import pysam
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from . import utils



CPG_MODS = [("C", 0, "m")]
M6A_MODS = [("A", 0, "a"), ("T", 1, "a")]
D_TYPE = np.int64


# =============================================================================
# CLASSES
# =============================================================================

class FiberView(ad.AnnData):
    # OLD constructor, do not use for new stuff
    def __init__(self, alignment_file, df, window=(-1000, 1000), 
                 min_mod_score=220, mark_cpgs=True, fully_span=True, 
                 region_interval=30, filter_args={'dist':3000, 'cutoff':2},
                 tags=['np', 'ec', 'rq'], fire=False, max_reads=300):
        row_anno_df_list = []
        seq_mtx_list = []
        m6a_mtx_list = []
        cpg_mtx_list = []
        nuc_pos_mtx_list = []
        nuc_len_mtx_list = []
        nuc_score_mtx_list = []
        msp_pos_mtx_list = []
        msp_len_mtx_list = []
        msp_score_mtx_list = []
        for i, row in df.iterrows():
            print(i)
            reads = utils.ReadList().get_reads(alignment_file, 
                                         ref_pos=(row.seqid, row.pos, row.strand),
                                         max_reads=max_reads)
            if fully_span:
                reads.filter_by_window(window, inplace=True)
            if filter_args is not None:
                reads.filter_by_end_meth(dist=filter_args['dist'], cutoff=filter_args['cutoff'], inplace=True)
            if len(reads) == 0:
                continue
            row_anno_df_list.append(reads.build_anno_df(row, tags=tags))
            seq_mtx_list.append(reads.build_seq_array(window))
            m6a_mtx_list.append(reads.build_mod_array(window,
                                                      mod_type=M6A_MODS, sparse=True, 
                                                      score_cutoff=220))
            cpg_mtx_list.append(reads.build_mod_array(window,
                                                      mod_type=CPG_MODS, sparse=True, 
                                                      score_cutoff=220))
            
            reg_pos_mtx, reg_len_mtx, reg_score_mtx = \
                reads.build_sparse_region_array(window, tags=('ns', 'nl'), 
                                                interval=region_interval)
            nuc_pos_mtx_list.append(reg_pos_mtx)
            nuc_len_mtx_list.append(reg_len_mtx)
            nuc_score_mtx_list.append(reg_score_mtx)
            
            if fire:
                reg_pos_mtx, reg_len_mtx, reg_score_mtx = \
                    reads.build_sparse_region_array(window, tags=('as', 'al', 'aq'), 
                                                    interval=region_interval)
            else:
                reg_pos_mtx, reg_len_mtx, reg_score_mtx = \
                    reads.build_sparse_region_array(window, tags=('as', 'al'), 
                                                    interval=region_interval)

            msp_pos_mtx_list.append(reg_pos_mtx)
            msp_len_mtx_list.append(reg_len_mtx)
            msp_score_mtx_list.append(reg_score_mtx)
            
        super().__init__(
            obs=pd.concat(row_anno_df_list, ignore_index=True),
            var=pd.DataFrame({"pos" : np.arange(window[0], window[1])})
            )
        self.X = csr_matrix(self.shape) # empty matrix, needed for AnnData.to_memory()
        self.layers['seq'] = np.vstack(seq_mtx_list)
        self.layers['m6a'] = vstack(m6a_mtx_list).tocsr()
        self.layers['cpg'] = vstack(cpg_mtx_list).tocsr()
        self.layers['nuc_pos'] = vstack(nuc_pos_mtx_list).tocsr()
        self.layers['nuc_len'] = vstack(nuc_len_mtx_list).tocsr()
        self.layers['nuc_score'] = vstack(nuc_score_mtx_list).tocsr()
        self.layers['msp_pos'] = vstack(msp_pos_mtx_list).tocsr()
        self.layers['msp_len'] = vstack(msp_len_mtx_list).tocsr()
        self.layers['msp_score'] = vstack(msp_score_mtx_list).tocsr()
        # add unstructured data
        self.uns['region_report_interval'] = region_interval
        self.uns['is_agg'] = False
        self.uns['region_base_names'] = ['nuc', 'msp']
        self.uns['mods'] = ['m6a', 'cpg']
        self.uns['bin_width'] = 1
        if mark_cpgs:
            self.mark_cpg_sites()
        
    def mark_cpg_sites(self, sparse=True):
        # make a new layer 'cpg_sites' with cpg sites as True, 
        # Known issue: all Cs at end of sequence are marked as not CpGs 
        cpg_sites = np.logical_and(self.layers['seq'][:, 0:-1] == b'C', 
                                   self.layers['seq'][:, 1:] == b'G')
        cpg_sites = np.pad(cpg_sites, pad_width=((0,0),(0,1)), mode='constant')
        if sparse:
            cpg_sites = csr_matrix(cpg_sites)
        self.layers['cpg_sites'] = cpg_sites
        return(None)
    def get_seq_records(self, id_col="read_name"):
        seqs = [Seq(bytes(row)) for row in self.layers['seq']]
        seq_records = []
        for i in range(self.shape[0]):
            seq_records.append( SeqRecord(seqs[i], id=self.obs[id_col][i], 
                                          description=self.obs.index[i]) )
        return(seq_records)

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


def build_fview(alignment_file, sites_df, window=(-1000, 1000), fully_span=True, 
             region_interval=30, filter_args={'dist':3000, 'cutoff':2},
             tags=['np', 'ec', 'rq'], fire=False, max_reads=300):
    
    row_anno_df_list = []
    seq_mtx_list = []
    m6a_mtx_list = []
    cpg_mtx_list = []
    nuc_pos_mtx_list = []
    nuc_len_mtx_list = []
    nuc_score_mtx_list = []
    msp_pos_mtx_list = []
    msp_len_mtx_list = []
    msp_score_mtx_list = []
    for i, row in sites_df.iterrows():
        print(i)
        reads = utils.ReadList().get_reads(alignment_file, 
                                     ref_pos=(row.seqid, row.pos, row.strand),
                                     max_reads=max_reads)
        if fully_span:
            reads.filter_by_window(window, inplace=True)
        if filter_args is not None:
            reads.filter_by_end_meth(dist=filter_args['dist'], cutoff=filter_args['cutoff'], inplace=True)
        if len(reads) == 0:
            continue
        row_anno_df_list.append(reads.build_anno_df(row, tags=tags))
        seq_mtx_list.append(reads.build_seq_array(window))
        m6a_mtx_list.append(reads.build_mod_array_from_def(window,
                                                  mod_def=PB_FS_mod_defs[0], sparse=True))
        cpg_mtx_list.append(reads.build_mod_array_from_def(window,
                                                  mod_def=PB_FS_mod_defs[1], sparse=True))
        
        reg_pos_mtx, reg_len_mtx, reg_score_mtx = \
            reads.build_sparse_region_array(window, tags=('ns', 'nl'), 
                                            interval=region_interval)
        nuc_pos_mtx_list.append(reg_pos_mtx)
        nuc_len_mtx_list.append(reg_len_mtx)
        nuc_score_mtx_list.append(reg_score_mtx)
        
        if fire:
            reg_pos_mtx, reg_len_mtx, reg_score_mtx = \
                reads.build_sparse_region_array(window, tags=('as', 'al', 'aq'), 
                                                interval=region_interval)
        else:
            reg_pos_mtx, reg_len_mtx, reg_score_mtx = \
                reads.build_sparse_region_array(window, tags=('as', 'al'), 
                                                interval=region_interval)

        msp_pos_mtx_list.append(reg_pos_mtx)
        msp_len_mtx_list.append(reg_len_mtx)
        msp_score_mtx_list.append(reg_score_mtx)
        
    fview = ad.AnnData(
        obs=pd.concat(row_anno_df_list),
        var=pd.DataFrame({"pos" : np.arange(window[0], window[1])})
        )
    fview.X = csr_matrix(fview.shape) # empty matrix, needed for AnnData.to_memory()
    fview.layers['seq'] = np.vstack(seq_mtx_list)
    fview.layers['m6a'] = vstack(m6a_mtx_list).tocsr()
    fview.layers['cpg'] = vstack(cpg_mtx_list).tocsr()
    fview.layers['nuc_pos'] = vstack(nuc_pos_mtx_list).tocsr()
    fview.layers['nuc_len'] = vstack(nuc_len_mtx_list).tocsr()
    fview.layers['nuc_score'] = vstack(nuc_score_mtx_list).tocsr()
    fview.layers['msp_pos'] = vstack(msp_pos_mtx_list).tocsr()
    fview.layers['msp_len'] = vstack(msp_len_mtx_list).tocsr()
    fview.layers['msp_score'] = vstack(msp_score_mtx_list).tocsr()
    # add unstructured data
    fview.uns['region_report_interval'] = region_interval
    fview.uns['is_agg'] = False
    fview.uns['region_base_names'] = ['nuc', 'msp']
    fview.uns['mods'] = ['m6a', 'cpg']
    fview.uns['bin_width'] = 1
    
    return(fview)


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



