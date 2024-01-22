import os
import pandas as pd

from src.utils import config as cfg

class MMseq2Runner:
    bin_path = cfg.MMSEQ2_BIN
    
    @staticmethod
    def csv_to_FASTA(f_in:str, f_out:str, unique_column:str='prot_id'):
        # converts a csv of proteins with codes to a FASTA file with codes as headers
        # column of csv is codes,..., prot_seq
        df = pd.read_csv(f_in, index_col=0)
        # get unique codes
        codes = df[unique_column].drop_duplicates().index
        # write to fasta
        with open(f_out, 'w') as f:
            for code in codes:
                seq = df.loc[code, "prot_seq"]
                if not isinstance(seq, str):
                    seq = seq[0]
                f.write(f'>{code}\n{seq}\n')
        
    @staticmethod
    def run_simple_clustering(fasta_in:str, out_dir:str, tmp_dir:str=None, 
                              force_overwrite:bool=False, verbose:bool=False):
        # --------------------------------------------
        # mmseqs createdb XY.fasta davisDB/DB
        # mmseqs cluster -s 16.0 --cov-mode 5 -c 0.6 davisDB/DB clu/DB tmp/
        # mmseqs createtsv davisDB/DB davisDB/DB clu/DB tsvs/davisDB_4sens_9c_5cov.tsv
        # --------------------------------------------
        
        # runs mmseq2 simple clustering on a fasta file
        # fasta_in: path to fasta file
        # out_dir: path to output directory
        # tmp_dir: path to tmp directory
        # force_overwrite: if True, will overwrite existing files
        # verbose: if True, will print mmseq2 output
        # returns: path to clustering results

        # check if output files already exist
        if not force_overwrite:
            if os.path.exists(f'{fasta_in}.clu_rep.faa'):
                print(f'Clustering results already exist for {fasta_in}. Skipping...')
                return f'{fasta_in}.clu_rep.faa.clstr'
        
        # make tmp dir if it doesn't exist
        if tmp_dir is not None:
            os.makedirs(tmp_dir, exist_ok=True)
            
        # create mmseq2 database
        cmd = f'{MMseq2Runner.bin_path} createdb {fasta_in} {fasta_in}.db'
        if verbose:
            print(cmd)
        os.system(cmd)
        
        # run mmseqs2 clustering
        
        # to get fewer clusters we change the following:
        # set --cov-mode to 1 for coverage of the target (note that 1 and 2 are the same in this case since our target is the same as the query)
        # set --cov-mode to 5 instead? "short seq. needs to be at least x% of other sequence"
        # decrease -c from 0.8 to 0.6 (this controls the proportion that the alignement needs to match)
        #
        
        cmd = f'{MMseq2Runner.bin_path} easy-cluster {fasta_in}.db {out_dir} {tmp_dir}'
        if verbose:
            print(cmd)
        os.system(cmd)
        
        # create tsv file
        
        return f'{fasta_in}.clu_rep.faa.clstr'
    
    @staticmethod
    def read_clusttsv_output(tsv_path:str):
        # reads the output of mmseq2 createtsv
        # returns a dict of {cluster_rep: [members], ...}
        
        # read tsv
        df = pd.read_csv(tsv_path, sep='\t', header=None)
        # rename cols
        df.columns = ['rep', 'member']
        
        return df.groupby('rep')['member'].apply(list).to_dict()