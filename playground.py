# %% building test datasets 
import logging
from src.data_prep.init_dataset import create_datasets
from src import config as cfg

logging.getLogger().setLevel(logging.INFO)

overwrite = True

create_datasets([cfg.DATA_OPT.PDBbind], [cfg.PRO_FEAT_OPT.nomsa],
                [cfg.PRO_EDGE_OPT.aflow_ring3],
                data_root=cfg.DATA_ROOT, overwrite=False,
                k_folds=5)

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
    if not os.path.exists(dst):
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
from src.utils.residue import Chain
alphaflow_dir = "../data/pdbbind/alphaflow_io/out_pdb_MD-distilled/"

invalid_files = {}
# removed = {}
for file in tqdm(os.listdir(alphaflow_dir)):
    if not file.endswith('.pdb'):
        continue
    file_path = os.path.join(alphaflow_dir, file)
    try:
        occurrences = Chain.get_model_count(file_path)
        if occurrences < 50:
            invalid_files[file] = (occurrences, file_path)
            
        if occurrences == 5: # delete and flag
            print(file_path)
            # os.remove(file_path)
            # removed[file] = file_path
            
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
    
# %%
