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
    def __init__(self, alignment_file, df, window=(-1000, 1000), 
                 min_mod_score=220, mark_cpgs=True, fully_span=True, 
                 region_interval=30):
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
                                         ref_pos=(row.seqid, row.pos, row.strand))
            if fully_span:
                reads.filter_by_window(window, inplace=True)
            if len(reads) == 0:
                continue
            row_anno_df_list.append(reads.build_anno_df(row))
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
            
            reg_pos_mtx, reg_len_mtx, reg_score_mtx = \
                reads.build_sparse_region_array(window, tags=('as', 'al'), 
                                                interval=region_interval)          
            msp_pos_mtx_list.append(reg_pos_mtx)
            msp_len_mtx_list.append(reg_len_mtx)
            msp_score_mtx_list.append(reg_score_mtx)
            
        super().__init__(
            obs=pd.concat(row_anno_df_list),
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
        # self.obs['site_name'] = ["{}:{}({})".format(row.seqid, row.pos, row.strand) 
        #                           for i, row in self.obs.iterrows()]
        # add unstructured data
        self.uns['region_report_interval'] = region_interval
        self.uns['is_agg'] = False
        self.uns['region_base_names'] = ['nuc', 'msp']
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

    def summarize_by_obs(self, obs_col_name='site_name', cols_to_keep=[]):
        # creats a new AnnData object where each row is 
        # this is not a FiberView object, the layers are all different
        As = []
        Cs = []
        Gs = []
        Ts = []
        cpg_sites = []
        cpgs = []
        m6as = []
        new_obs_rows = []
        if not obs_col_name in cols_to_keep:
            cols_to_keep.append(obs_col_name)
        if not 'cpg_sites' in self.layers.keys():
            self.mark_cpg_sites()
        for obs_val in np.unique(self.obs[obs_col_name]):
            a_subset = self[self.obs[obs_col_name] == obs_val, :]
            As.append(np.sum(a_subset.layers['seq'] == b'A', axis=0))
            Cs.append(np.sum(a_subset.layers['seq'] == b'C', axis=0))
            Gs.append(np.sum(a_subset.layers['seq'] == b'G', axis=0))
            Ts.append(np.sum(a_subset.layers['seq'] == b'T', axis=0))
            cpg_sites.append(np.sum(a_subset.layers['cpg_sites'], axis=0))
            cpgs.append(np.sum(a_subset.layers['cpg'], axis=0))
            m6as.append(np.sum(a_subset.layers['m6a'], axis=0))
            # take the first row of subset.obs as new obs row
            new_obs_row = a_subset.obs[cols_to_keep].iloc[0].copy()
            new_obs_row['n_seqs'] = a_subset.shape[0]
            new_obs_rows.append(new_obs_row)
        new_adata = ad.AnnData(
            obs=pd.DataFrame(new_obs_rows, index=np.arange(len(new_obs_rows))) ,
            var=self.var
            )
        # added np.asarray() because I was getting np.matrix objects in the layers
        # for the dataset of all genes.
        new_adata.layers['m6a'] = np.asarray(np.vstack(m6as))
        new_adata.layers['cpg'] = np.asarray(np.vstack(cpgs))
        new_adata.layers['As'] = np.asarray(np.vstack(As))
        new_adata.layers['Cs'] = np.asarray(np.vstack(Cs))
        new_adata.layers['Gs'] = np.asarray(np.vstack(Gs))
        new_adata.layers['Ts'] = np.asarray(np.vstack(Ts))
        new_adata.layers['CpGs'] = np.asarray(np.vstack(cpg_sites))
        new_adata.uns = self.uns
        new_adata.uns['is_agg'] = True
        return(new_adata)
    def get_seq_records(self, id_col="read_name"):
        seqs = [Seq(bytes(row)) for row in self.layers['seq']]
        seq_records = []
        for i in range(self.shape[0]):
            seq_records.append( SeqRecord(seqs[i], id=self.obs[id_col][i], 
                                          description=self.obs.index[i]) )
        return(seq_records)



def ad2fv(ad_object):
    # converts an AnnData object to FiberView, in place and returns the object.
    assert type(ad_object) == ad.AnnData, \
           "expected object of type AnnData, got {}".format(type(ad_object)) 
    # add checks for expected fields/layers here
    ad_object.__class__ = FiberView
    return(ad_object)

def read_h5ad(filename, backed=None, as_sparse=(), 
              as_sparse_fmt=csr_matrix, 
              chunk_size=6000):
    # wrapper around anndata.read_h5ad to return a FiberView object
    adata = ad.read_h5ad(filename=filename,
                         backed=backed,
                         as_sparse=as_sparse,
                         as_sparse_fmt = as_sparse_fmt,
                         chunk_size=chunk_size)
    return(ad2fv(adata))

