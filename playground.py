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
# files are .pdb with 50 "models" in each
for file in tqdm(os.listdir(alphaflow_dir)):
    if not file.endswith('.pdb'):
        continue
    
    code, _ = os.path.splitext(file)
    pid = df_unique.loc[code].prot_id
    
    os.symlink(f"{alphaflow_dir}/{file}", 
               f"{ln_dir}/{pid}.pdb")
    

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
