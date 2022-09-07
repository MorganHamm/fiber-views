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
    def __init__(self, alignment_file, df, window_offset=1000, 
                 min_mod_score=220, mark_cpgs=True):
        row_anno_df_list = []
        seq_mtx_list = []
        m6a_mtx_list = []
        cpg_mtx_list = []
        for i, row in df.iterrows():
            reads = utils.ReadList().get_reads(alignment_file, 
                                         ref_pos=(row.seqid, row.pos, row.strand))
            reads.filter_by_window(window_offset, inplace=True)
            if len(reads) == 0:
                continue
            row_anno_df_list.append(reads.build_anno_df(row))
            seq_mtx_list.append(reads.build_seq_array(window_offset))
            m6a_mtx_list.append(reads.build_mod_array(window_offset,
                                                      mod_type=M6A_MODS, sparse=True, 
                                                      score_cutoff=220))
            cpg_mtx_list.append(reads.build_mod_array(window_offset,
                                                      mod_type=CPG_MODS, sparse=True, 
                                                      score_cutoff=220))
        super().__init__(
            obs=pd.concat(row_anno_df_list),
            var=pd.DataFrame({"pos" : np.arange(-window_offset, window_offset)})
            )
        self.layers['seq'] = np.vstack(seq_mtx_list)
        self.layers['m6a'] = vstack(m6a_mtx_list).tocsr()
        self.layers['cpg'] = vstack(cpg_mtx_list).tocsr()
        # self.obs['site_name'] = ["{}:{}({})".format(row.seqid, row.pos, row.strand) 
        #                           for i, row in self.obs.iterrows()]
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
        new_adata.layers['m6a'] = np.vstack(m6as)
        new_adata.layers['cpg'] = np.vstack(cpgs)
        new_adata.layers['As'] = np.vstack(As)
        new_adata.layers['Cs'] = np.vstack(Cs)
        new_adata.layers['Gs'] = np.vstack(Gs)
        new_adata.layers['Ts'] = np.vstack(Ts)
        new_adata.layers['CpGs'] = np.vstack(cpg_sites)
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

