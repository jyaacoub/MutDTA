import os, subprocess
from multiprocessing import Pool
import pandas as pd

from src.utils import config as cfg
from src.data_prep.processors import Processor

from tqdm import tqdm


class MSARunner(Processor):
    hhsuite_bin_dir = '/cluster/tools/software/centos7/hhsuite/3.3.0/bin'
    bin_hhblits = f'{hhsuite_bin_dir}/hhblits'
    bin_hhfilter = f'{hhsuite_bin_dir}/hhfilter'
    UniRef30_dir = '/cluster/projects/kumargroup/mslobody/Protein_Communities/01_MSA/databases/UniRef30_2020_06'

    @staticmethod    
    def hhblits(f_in:str, f_out:str, n_cpus=6, n_iter:int=2,
                bin_path:str=None, dataset:str=None) -> subprocess.CompletedProcess:
        # hhblits works on a single sequence at once
        # can pass FASTA file
        
        cmd = f"{bin_path or MSARunner.bin_hhblits}" + \
                f" -i {f_in}" + \
                f" -oa3m {f_out}" + \
                f" -d {dataset or MSARunner.UniRef30_dir}" + \
                f" -cpu {n_cpus}" + \
                f" -n {n_iter}"
        return subprocess.run(cmd, capture_output=True, check=True, shell=True)

    @staticmethod    
    def hhfilter(f_in: str, f_out:str=None, 
                 max_seq_id:int=90, bin_path:str=None) -> subprocess.CompletedProcess:
        assert '.a3m' in f_in, 'Input file must be in a3m format'
        
        f_out = f_in.split('.a3m')[0] + '_filtered.a3m' if f_out is None else f_out
        
        cmd = f'{bin_path or MSARunner.bin_hhfilter} -id {max_seq_id} -i {f_in} -o {f_out}'
        return subprocess.run(cmd, capture_output=True, check=True, shell=True)

    @staticmethod
    def clean_msa(f_in: str, f_out:str):
        # getting sequences from file
        with open(f_in, 'r') as f:
            lines = f.readlines()
            seq = [l for l in lines if l[0] != '>']
        
        # writing sequences to file without lowercase letters 
        # (hhblits adds lowercase letters to indicate insertions)
        with open(f_out, 'w') as f:
            for l in seq:
                # removing lowercase letters
                new_l = ''
                for c in l:
                    if c.isupper() or c == '-':
                        new_l += c
                f.write(new_l+'\n')

    @staticmethod
    def check_aln_lines(fp:str, limit=None):
        if not os.path.isfile(fp): return False
        
        with open(fp, 'r') as f:
            lines = f.readlines()
            seq_len = len(lines[0])
            limit = len(lines) if limit is None else limit
            
            for i,l in enumerate(lines[:limit]):
                if l[0] == '>' or len(l) != seq_len:
                    return False
        return True

    @staticmethod
    def process_msa(f_in:str, f_out:str, hhfilter_bin:str=None):
        MSARunner.hhfilter(f_in, f_out, bin_path=hhfilter_bin)
        # overwrites filtered msa with cleaned msa
        MSARunner.clean_msa(f_in=f_out, f_out=f_out)
        MSARunner.check_aln_lines(f_out)

    @staticmethod    
    def process_msa_dir(in_dir:str, out_dir:str=None,
                        postfix:str='.msa.a3m', hhfilter_bin:str=None):
        """
        Processes msas after hhblits has been run:
        1. Filters the MSA using hhfilter.
        2. Removes lowercase letters and '>' labels from the MSA and saves it as 
        '<pdb>_clean.a3m'.

        Parameters
        ----------
        `hhfilter_bin` : str
            binary path to hhfilter see: https://github.com/soedinglab/hh-suite
        `dir_p` : str
            directory path to the MSA files.
        `postfix` : str, optional
            postfix pattern for msa files, by default '.msa.a3m'
        """
        if out_dir is not None:
            os.makedirs(out_dir, exist_ok=True)
        else:
            out_dir = in_dir
        msas = [f for f in os.listdir(in_dir) if f.endswith(postfix)]
        
        for msa in tqdm(msas, 'Filtering and cleaning MSAs'):
            f_in = f'{in_dir}/{msa}'
            f_out = f'{out_dir}/{msa[:-len(postfix)]}.aln'
            if not os.path.isfile(f_out):
                MSARunner.process_msa(f_in, f_out, hhfilter_bin)

    @staticmethod    
    def process_msa_file(args):
        hhfilter_bin, f_in, f_out = args
        if not os.path.isfile(f_out):
            MSARunner.process_msa(f_in, f_out, hhfilter_bin)

    @staticmethod
    def multiprocess_msa_dir(in_dir:str, out_dir:str=None,postfix:str='.msa.a3m',
                            hhfilter_bin:str=None):
        if out_dir is not None:
            os.makedirs(out_dir, exist_ok=True)
        else:
            out_dir = in_dir
        msas = [f for f in os.listdir(in_dir) if f.endswith(postfix)]

        # Create a Pool for multiprocessing
        with Pool() as pool:
            args_list = [(hhfilter_bin, os.path.join(in_dir, msa), 
                        os.path.join(out_dir, 
                                    msa[:-len(postfix)]) + '.aln') for msa in msas]
            with tqdm(total=len(args_list), desc='Processing MSAs') as pbar:
                for _ in pool.imap_unordered(MSARunner.process_msa_file, args_list):
                    pbar.update(1)

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
    
    
    
if __name__ == '__main__':
    from src.data_prep.datasets import BaseDataset
    csv = '/cluster/home/t122995uhn/projects/data/PlatinumDataset/nomsa_binary/full/XY.csv'
    df = pd.read_csv(csv, index_col=0)
    #################### Get unique proteins:
    unique_df = BaseDataset.get_unique_prots(df)
    
    ########################## Get job partition
    num_arrays = 100
    array_idx = 0#${SLURM_ARRAY_TASK_ID}
    partition_size = len(unique_df) / num_arrays
    start, end = int(array_idx*partition_size), int((array_idx+1)*partition_size)
    
    unique_df = unique_df[start:end]
    
    raw_dir = '/cluster/home/t122995uhn/projects/data/PlatinumDataset/raw'

    #################################### create fastas
    fa_dir = os.path.join(raw_dir, 'platinum_fa')
    os.makedirs(fa_dir, exist_ok=True)
    MSARunner.csv_to_fasta_dir(csv_or_df=unique_df, out_dir=fa_dir)

    ##################################### Run hhblits
    aln_dir = os.path.join(raw_dir, 'platinum_aln')
    os.makedirs(aln_dir, exist_ok=True)

    # finally running
    for _, (prot_id, pro_seq) in tqdm(
                    unique_df[['prot_id', 'prot_seq']].iterrows(), 
                    desc='Running hhblits',
                    total=len(unique_df)):
        in_fp = os.path.join(fa_dir, f"{prot_id}.fasta")
        out_fp = os.path.join(aln_dir, f"{prot_id}.a3m")
        
        if not os.path.isfile(out_fp):
            MSARunner.hhblits(in_fp, out_fp)