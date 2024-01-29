"""
Adapted from https://github.com/HWaymentSteele/AF_Cluster/blob/main/scripts/ClusterMSA.py
"""

import numpy as np
from Bio import SeqIO
import logging as log

def load_fasta(fil):
    # check to see if there are '>' in file that represent a file with IDs
    with open(fil, 'r') as f:
        first_line = f.readline()
        has_ids = '>' == first_line[0]
        
    if has_ids:
        seqs, IDs =[], []
        with open(fil) as handle:
            for record in SeqIO.parse(handle, "fasta"):
                seq = ''.join([x for x in record.seq])
                IDs.append(record.id)
                seqs.append(seq)
        return IDs, seqs
    
    log.warning('MSA is not in FASTA format so assuming the file is just a list of sequences')
    with open(fil, 'r') as f:
        seqs = []
        for line in f.readlines():
            if line[0] == '>': raise SyntaxError("Invalid MSA input format, can't detect what input is.")
            if len(line) <= 5: continue # new line or noise
            seqs.append(line.strip()) # strip any newline or spaces
        return list(range(len(seqs))), seqs

def write_fasta(names, seqs, outfile='tmp.fasta'):
        with open(outfile,'w') as f:
                for nm, seq in list(zip(names, seqs)):
                        f.write(">%s\n%s\n" % (nm, seq))

def consensusVoting(seqs):
    ## Find the consensus sequence
    consensus = ""
    residues = "ACDEFGHIKLMNPQRSTVWY-"
    n_chars = len(seqs[0])
    for i in range(n_chars):
        baseArray = [x[i] for x in seqs]
        baseCount = np.array([baseArray.count(a) for a in list(residues)])
        vote = np.argmax(baseCount)
        consensus += residues[vote]

    return consensus

def encode_seqs(seqs, max_len=108, alphabet=None):
    
    if alphabet is None:
        alphabet = "ACDEFGHIKLMNPQRSTVWY-"
    
    arr = np.zeros([len(seqs), max_len, len(alphabet)])
    for j, seq in enumerate(seqs):
        for i,char in enumerate(seq):
            for k, res in enumerate(alphabet):
                if char==res:
                    arr[j,i,k]+=1
    return arr.reshape([len(seqs), max_len*len(alphabet)])

import os
import pandas as pd
from Bio import SeqIO
import numpy as np

from polyleven import levenshtein
from sklearn.cluster import DBSCAN

import matplotlib.pyplot as plt
import seaborn as sns

class AF_Clust:
    def __init__(self, keyword:str, 
                 input_msa:str, 
                 output_dir:str, 
                 
                 # DBSCAN ARGUMENTS:
                 eps_val:float=None, 
                 min_eps:int=3,
                 max_eps:int=20,
                 eps_step:float=0.5,
                 min_samples:int=3,
                 
                 n_controls:int=0,      # number of control msas to generate
                 gap_cutoff:float=0.25,  # limit for gaps in seq (removed if larger)
                 
                 verbose:bool=False,
                 scan:bool=False,
                 resample:bool=False,
                 
                 
                 run_PCA:bool=False,
                 run_TSNE:bool=False,
                 ) -> None:
        
        self.keyword = keyword
        self.i       = input_msa
        self.o       = output_dir
        
        self.n_controls  = n_controls
        self.verbose     = verbose
        self.scan        = scan
        
        self.resample    = resample
        self.gap_cutoff  = gap_cutoff
        
        self.eps_val     = eps_val
        self.min_eps     = min_eps
        self.max_eps     = max_eps
        self.eps_step    = eps_step
        self.min_samples = min_samples
        
        self.run_PCA     = run_PCA
        self.run_TSNE    = run_TSNE
        
        # SETUP LOAD MSA and PREPARE DIR
        os.makedirs(self.o, exist_ok=True)
        IDs, seqs = load_fasta(self.i)

        seqs = [''.join([x for x in s if x.isupper() or x=='-']) for s in seqs] # remove lowercase letters in alignment

        self.df = pd.DataFrame({'SequenceName': IDs, 'sequence': seqs})

        query_ = self.df.iloc[:1]
        self.df = self.df.iloc[1:]

        if self.resample:
            self.df = self.df.sample(frac=1)

        L = len(self.df.sequence.iloc[0])
        N = len(self.df)

        self.df['frac_gaps'] = [x.count('-')/L for x in self.df['sequence']]

        former_len=len(self.df)
        self.df = self.df.loc[self.df.frac_gaps<self.gap_cutoff]

        new_len=len(self.df)
        log.info(self.keyword)
        log.info("%d seqs removed for containing more than %d%% gaps, %d remaining" % (former_len-new_len, int(self.gap_cutoff*100), new_len))
        ohe_seqs = encode_seqs(self.df.sequence.tolist(), max_len=L)

        n_clusters=[]
        eps_test_vals=np.arange(self.min_eps, self.max_eps+self.eps_step, self.eps_step)
        
        # PERFORM SCAN
        if self.eps_val is None: # performing scan
            log.debug('eps\tn_clusters\tn_not_clustered')

            for eps in eps_test_vals:

                testset = encode_seqs(self.df.sample(frac=0.25).sequence.tolist(), max_len=L)
                clustering = DBSCAN(eps=eps, min_samples=self.min_samples).fit(testset)
                n_clust = len(set(clustering.labels_))
                n_not_clustered = len(clustering.labels_[np.where(clustering.labels_==-1)])
                log.debug('%.2f\t%d\t%d' % (eps, n_clust, n_not_clustered))
                n_clusters.append(n_clust)
                if eps >= 15 and n_clust==1:
                    break

            eps_to_select = eps_test_vals[np.argmax(n_clusters)]
        else:
            eps_to_select = self.eps_val
            
        # PERFORM CLUSTERING:
        cluster_metadata = self.cluster(L, self.df, query_, ohe_seqs, eps_to_select)
        
        if self.run_PCA:
            self.pca(self.df, query_)
        if self.run_TSNE:
            self.tsne(self.df, query_)
        
        outfile = self.o+"/"+self.keyword+'_clustering_assignments.tsv'
        log.info('wrote clustering data to %s' % outfile)
        self.df.to_csv(outfile,index=False, sep='\t')

        metad_outfile = self.o+"/"+self.keyword+'_cluster_metadata.tsv'
        log.info('wrote cluster metadata to %s' % metad_outfile)
        metad_df = pd.DataFrame.from_records(cluster_metadata)
        metad_df.to_csv(metad_outfile,index=False, sep='\t')
        
    def cluster(self, L, df, query_, ohe_seqs, eps_to_select):
        clustering = DBSCAN(eps=eps_to_select, min_samples=self.min_samples).fit(ohe_seqs)

        log.info('Selected eps=%.2f' % eps_to_select)

        log.info("%d total seqs" % len(df))

        df['dbscan_label'] = clustering.labels_

        clusters = [x for x in df.dbscan_label.unique() if x>=0]
        unclustered = len(df.loc[df.dbscan_label==-1])

        log.info('%d clusters, %d of %d not clustered (%.2f)' % (len(clusters), unclustered, len(df), unclustered/len(df)))

        avg_dist_to_query = np.mean([1-levenshtein(x, query_['sequence'].iloc[0])/L for x in df.loc[df.dbscan_label==-1]['sequence'].tolist()])
        log.info('avg identity to query of unclustered: %.2f' % avg_dist_to_query)

        avg_dist_to_query = np.mean([1-levenshtein(x, query_['sequence'].iloc[0])/L for x in df.loc[df.dbscan_label!=-1]['sequence'].tolist()])
        log.info('avg identity to query of clustered: %.2f' % avg_dist_to_query)
        
        cluster_metadata=[]
        for clust in clusters:
            tmp = df.loc[df.dbscan_label==clust]

            cs = consensusVoting(tmp.sequence.tolist())

            avg_dist_to_cs = np.mean([1-levenshtein(x,cs)/L for x in tmp.sequence.tolist()])
            avg_dist_to_query = np.mean([1-levenshtein(x,query_['sequence'].iloc[0])/L for x in tmp.sequence.tolist()])

            if self.verbose:
                log.debug('Cluster %d consensus seq, %d seqs:' % (clust, len(tmp)))
                log.debug(cs)
                log.debug('#########################################')
                for _, row in tmp.iterrows():
                    log.debug(row['SequenceName'], row['sequence'])
                log.debug('#########################################')

            tmp = pd.concat([query_, tmp], axis=0)

            cluster_metadata.append({'cluster_ind': clust, 'consensusSeq': cs, 'avg_lev_dist': '%.3f' % avg_dist_to_cs, 
                'avg_dist_to_query': '%.3f' % avg_dist_to_query, 'size': len(tmp)})

            write_fasta(tmp.SequenceName.tolist(), tmp.sequence.tolist(), outfile=self.o+'/'+self.keyword+'_'+"%03d" % clust+'.a3m')

        log.info(f'writing {self.n_controls} size-10 uniformly sampled clusters')
        for i in range(self.n_controls):
            tmp = df.sample(n=10)
            tmp = pd.concat([query_, tmp], axis=0)
            write_fasta(tmp.SequenceName.tolist(), tmp.sequence.tolist(), outfile=self.o+'/'+self.keyword+'_U10-'+"%03d" % i +'.a3m') 
        if len(df)>100:
            log.info(f'writing {self.n_controls} size-100 uniformly sampled clusters')
            for i in range(self.n_controls):
                tmp = df.sample(n=100)
                tmp = pd.concat([query_, tmp], axis=0)
                write_fasta(tmp.SequenceName.tolist(), tmp.sequence.tolist(), outfile=self.o+'/'+self.keyword+'_U100-'+"%03d" % i +'.a3m')

        return cluster_metadata
        
    def pca(self, df, query_):
        from sklearn.decomposition import PCA
        print('Running PCA ...')
        ohe_vecs = encode_seqs(df.sequence.tolist(), max_len=L)
        mdl = PCA()
        embedding = mdl.fit_transform(ohe_vecs)

        query_embedding = mdl.transform(encode_seqs(query_.sequence.tolist(), max_len=L))

        df['PC 1'] = embedding[:,0]
        df['PC 2'] = embedding[:,1]

        query_['PC 1'] = query_embedding[:,0]
        query_['PC 2'] = query_embedding[:,1]

        self.plot_landscape('PC 1', 'PC 2', df, query_, 'PCA')

        print('Saved PCA plot to '+self.o+"/"+self.keyword+'_PCA.pdf')

    def tsne(self, df, query_):
        from sklearn.manifold import TSNE
        print('Running TSNE ...')
        ohe_vecs = encode_seqs(df.sequence.tolist()+[query_.sequence.tolist()], max_len=L)
        # different than PCA because tSNE doesn't have .transform attribute

        mdl = TSNE()
        embedding = mdl.fit_transform(ohe_vecs)

        df['TSNE 1'] = embedding[:-1,0]
        df['TSNE 2'] = embedding[:-1,1]

        query_['TSNE 1'] = embedding[-1:,0]
        query_['TSNE 2'] = embedding[-1:,1]

        self.plot_landscape('TSNE 1', 'TSNE 2', df, query_, 'TSNE')

        print('Saved TSNE plot to '+self.o+"/"+self.keyword+'_TSNE.pdf')

    def plot_landscape(self, x, y, df, query_, plot_type):

        plt.figure(figsize=(5,5))
        tmp = df.loc[df.dbscan_label==-1]
        plt.scatter(tmp[x], tmp[y], color='lightgray', marker='x', label='unclustered')

        tmp = df.loc[df.dbscan_label>9]
        plt.scatter(tmp[x],tmp[y], color='black', label='other clusters')

        tmp = df.loc[df.dbscan_label>=0][df.dbscan_label<=9]
        sns.scatterplot(x=x,y=y, hue='dbscan_label', data=tmp, palette='tab10',linewidth=0)

        plt.scatter(query_[x],query_[y], color='red', marker='*', s=150, label='Ref Seq')
        plt.legend(bbox_to_anchor=(1,1), frameon=False)

        plt.xlabel(x)
        plt.ylabel(y)
        plt.tight_layout()

        plt.savefig(self.o+"/"+self.keyword+'_'+plot_type+'.pdf', bbox_inches='tight')

if __name__ == "__main__":    
    dir_p = f"/cluster/home/t122995uhn/projects/colabfold"
    pid = '1a1e'
    msa = f"{dir_p}/pdbbind_a3m/{pid}.msa.a3m"
    af = AF_Clust(keyword="test-"+pid, input_msa=msa, output_dir=f"{dir_p}/test_af_clust/")
