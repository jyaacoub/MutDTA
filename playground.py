#%%

chain_sequence_validation_start = time.time()
assert target_seq is None or chains[0].getSequence() == target_seq, \
    f'Target seq is not chain seq for {pdb_fp} ({af_confs[0]})'
logging.debug(f"Chain sequence validation: {time.time() - chain_sequence_validation_start} seconds")

contact_map_start = time.time()
M = np.array([c.get_contact_map() for c in chains]) < 8.0
dist_cmap = np.sum(M, axis=0) / len(M)
logging.debug(f"Contact map calculation: {time.time() - contact_map_start} seconds")

ring3_runner_start = time.time()
input_pdb, files = Ring3Runner.run(af_confs, overwrite=False)
seq_len = len(Chain(input_pdb))
logging.debug(f"Ring3Runner run: {time.time() - ring3_runner_start} seconds")

ring3_cmaps_build_start = time.time()
r3_cmaps = []
for k, fp in files.items():
    cmap = Ring3Runner.build_cmap(fp, seq_len)
    r3_cmaps.append(cmap)
logging.debug(f"Ring3 cmaps build: {time.time() - ring3_cmaps_build_start} seconds")

all_cmaps_transpose_start = time.time()
all_cmaps = np.array(r3_cmaps + [dist_cmap], dtype=np.float32)  # [6, L, L]
all_cmaps = all_cmaps.transpose(1, 2, 0)  # [L, L, 6]
logging.debug(f"All cmaps transpose: {time.time() - all_cmaps_transpose_start} seconds")

logging.debug(f'Total runtime: {time.time() - start_time} seconds')
logging.debug(f'Ring3Runner all_cmaps: {all_cmaps.shape}')


# %% test XY.csv
from src.utils.residue import Chain
m = Chain.get_all_models('/cluster/home/t122995uhn/projects/data/pdbbind/alphaflow_io/out_pid_ln/T2I3Q3.pdb')


# %% building test datasets 
import logging
from src.data_prep.datasets import PDBbindDataset
from src import config as cfg

logging.getLogger().setLevel(logging.INFO)

data_root = cfg.DATA_ROOT
overwrite = True
FEATURE = cfg.PRO_FEAT_OPT.nomsa
EDGE = cfg.PRO_EDGE_OPT.ring3 # cfg.PRO_EDGE_OPT.simple # simple edges (freq of contact) 

ligand_feature = cfg.LIG_FEAT_OPT.original
ligand_edge = cfg.LIG_EDGE_OPT.binary

# prev was from subsampled msa (f'{data_root}/pdbbind/pdbbind_af2_out/all_ln/'):
af_conf_dir = f"{data_root}/pdbbind/alphaflow_io/out_pid_ln/" 


dataset = PDBbindDataset(
        save_root=f'{data_root}/PDBbindDataset/alphaflow/',
        data_root=f'{data_root}/pdbbind/v2020-other-PL/',
        aln_dir=None, # not relevant for nomsa
        cmap_threshold=8.0,
        overwrite=overwrite, # overwrite old cmap.npy files
        af_conf_dir=af_conf_dir,
        feature_opt=FEATURE,
        edge_opt=EDGE,
        ligand_feature=ligand_feature,
        ligand_edge=ligand_edge,
        alphaflow=True
        )


#%%
from src.data_prep.datasets import BaseDataset
import pandas as pd
csv_p = "/cluster/home/t122995uhn/projects/data/PDBbindDataset/nomsa_ring3_original_binary/full/XY.csv"

df = pd.read_csv(csv_p, index_col=0)
df_unique = BaseDataset.get_unique_prots(df)

# %%
import os
from tqdm import tqdm
alphaflow_dir = "/cluster/home/t122995uhn/projects/data/pdbbind/alphaflow_io/out_pdb_MD-distilled/"
ln_dir =        "/cluster/home/t122995uhn/projects/data/pdbbind/alphaflow_io/out_pid_ln/"

os.makedirs(ln_dir, exist_ok=True)

# First, check and remove any broken links in the destination directory
for link_file in tqdm(os.listdir(ln_dir), desc="Checking for broken links"):
    ln_p = os.path.join(ln_dir, link_file)
    if os.path.islink(ln_p) and not os.path.exists(ln_p):
        print(f"Removing broken link: {ln_p}")
        os.remove(ln_p)


# %% files are .pdb with 50 "models" in each
for file in tqdm(os.listdir(alphaflow_dir)):
    if not file.endswith('.pdb'):
        continue
    
    code, _ = os.path.splitext(file)
    pid = df_unique.loc[code].prot_id
    src, dst = f"{alphaflow_dir}/{file}", f"{ln_dir}/{pid}.pdb"
    if os.path.exists(dst): 
        os.remove(dst)
    os.symlink(src,dst)
    

# %% RUN RING3
# %% Run RING3 on finished confirmations from AlphaFlow
from src.utils.residue import Ring3Runner

files = [os.path.join(ln_dir, f) for f in \
            os.listdir(ln_dir) if f.endswith('.pdb')]

Ring3Runner.run_multiprocess(pdb_fps=files)


# %% checking the number of models in each file, flagging any issues:
from tqdm import tqdm
import os
alphaflow_dir = "../data/pdbbind/alphaflow_io/out_pdb_MD-distilled/"

invalid_files = {}
# removed = {}
for file in tqdm(os.listdir(alphaflow_dir)):
    if not file.endswith('.pdb'):
        continue
    file_path = os.path.join(alphaflow_dir, file)
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        occurrences = content.count("MODEL")
        if occurrences < 50:
            invalid_files[file] = (occurrences, file_path)
            
        if occurrences == 5: # delete and flag
            print(file_path)
            # os.remove(file_path)
            # removed[file] = file_path
            
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
    
# %%
