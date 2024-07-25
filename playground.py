#%%
import os
import logging
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
from src.utils.residue import Chain
from multiprocessing import Pool, cpu_count
from src.data_prep.datasets import BaseDataset

# df = pd.read_csv("/cluster/home/t122995uhn/projects/MutDTA/df_base.csv", index_col=0)
df = pd.read_csv("/cluster/home/t122995uhn/projects/MutDTA/df_base_filtered.csv", index_col=0)
logging.getLogger().setLevel(logging.DEBUG)

def process_protein_multiprocessing(args):
    """
    Checks if protein has conf file and correct sequence, returns:
        - None, None - if it has a conf file and is correct
        - pid, None - is missing a conf file
        - pid, seq - has a conf file but is not correct sequence.
    """
    group_codes, code, pid, seq, af_conf_dir, is_pdbbind, files = args
    MIN_MODEL_COUNT = 5
    
    correct_seq = False
    matching_code = None
    af_confs = []
    if is_pdbbind:
        for c in group_codes:
            af_fp = os.path.join(af_conf_dir, f'{c}.pdb')
            if os.path.exists(af_fp):
                af_confs = [af_fp]
                matching_code = code
                if Chain(af_fp).sequence == seq:
                    correct_seq = True
                    break
        
    else:
        af_confs = [os.path.join(af_conf_dir, f) for f in files if f.startswith(pid)]
    
    if len(af_confs) == 0:
        return pid, None
    
    # either all models in one pdb file (alphaflow) or spread out across multiple files (AF2 msa subsampling)
    model_count = len(af_confs) if len(af_confs) > 1 else 5# Chain.get_model_count(af_confs[0])
    
    if model_count < MIN_MODEL_COUNT:
        return pid, None
    elif not correct_seq: # final check
        af_seq = Chain(af_confs[0]).sequence
        if seq != af_seq:
            logging.debug(f'Mismatched sequence for {pid}')
            # if matching_code == code: # something wrong here -> incorrect seq but for the right code?
            #     return pid, af_seq
            return pid, matching_code
        
    if matching_code != code:
        return None, matching_code
    return None, None

#%% check_missing_confs method
af_conf_dir:str = '/cluster/home/t122995uhn/projects/data/pdbbind/alphaflow_io/out_pdb_MD-distilled/'
is_pdbbind=True

df_unique:pd.DataFrame = df.drop_duplicates('prot_id')
df_pid_groups = df.groupby(['prot_id']).groups

missing = set()
mismatched = {}
# total of 3728 unique proteins with alphaflow confs (named by pdb ID)
files = None
if not is_pdbbind:
    files = [f for f in os.listdir(af_conf_dir) if f.endswith('.pdb')]

with Pool(processes=cpu_count()) as pool:
    tasks = [(df_pid_groups[pid], code, pid, seq, af_conf_dir, is_pdbbind, files) \
                    for code, (pid, seq) in df_unique[['prot_id', 'prot_seq']].iterrows()]

    for pid, new_seq in tqdm(pool.imap_unordered(process_protein_multiprocessing, tasks), 
                    desc='Filtering out proteins with missing PDB files for multiple confirmations', 
                    total=len(tasks)):
        if new_seq is not None:
            mismatched[pid] = new_seq
        elif pid is not None: # just pid -> missing af files
            missing.add(pid)

print(len(missing),len(mismatched))

#%% make subsitutions for rows
df = pd.read_csv("/cluster/home/t122995uhn/projects/MutDTA/df_base.csv", index_col=0)
df_mismatched = pd.DataFrame.from_dict(mismatched, orient='index', columns=['code'])
df_mismatched_sub = df.loc[df_mismatched['code']][['prot_id', 'prot_seq']].reset_index()
df_mismatched = df_mismatched.merge(df_mismatched_sub, on='code')

df_mismatched = df_mismatched.merge(df_mismatched_sub, on='code')
dff = pd.read_csv("/cluster/home/t122995uhn/projects/MutDTA/df_base_filtered.csv")
dffm = dff.merge(df_mismatched, on='code')
#%%
from src.data_prep.datasets import BaseDataset
import pandas as pd
csv_p = "/cluster/home/t122995uhn/projects/data/PDBbindDataset/nomsa_binary_original_binary/full/XY.csv"

df = pd.read_csv(csv_p, index_col=0)

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
    pid = df.loc[code].prot_id
    src, dst = f"{alphaflow_dir}/{file}", f"{ln_dir}/{pid}.pdb"
    if not os.path.exists(dst):
        os.symlink(src,dst)

#%%
########################################################################
########################## BUILD DATASETS ##############################
########################################################################
import os
from src.data_prep.init_dataset import create_datasets
from src import cfg
import logging
cfg.logger.setLevel(logging.DEBUG)

splits = '/cluster/home/t122995uhn/projects/MutDTA/splits/davis/'
create_datasets(cfg.DATA_OPT.davis, 
                feat_opt=cfg.PRO_FEAT_OPT.nomsa, 
                edge_opt=[cfg.PRO_EDGE_OPT.aflow, cfg.PRO_EDGE_OPT.binary],
                ligand_features=[cfg.LIG_FEAT_OPT.original, cfg.LIG_FEAT_OPT.gvp], 
                ligand_edges=cfg.LIG_EDGE_OPT.binary, overwrite=True,
                k_folds=5, 
                test_prots_csv=f'{splits}/test.csv',
                val_prots_csv=[f'{splits}/val{i}.csv' for i in range(5)],
                data_root=os.path.abspath('../data/test/'))


# %%
########################################################################
########################## VIOLIN PLOTTING #############################
########################################################################
import logging
from matplotlib import pyplot as plt

from src.analysis.figures import prepare_df, custom_fig, fig_combined

models = {
    'DG': ('nomsa', 'binary', 'original', 'binary'),
    'esm': ('ESM', 'binary', 'original', 'binary'), # esm model
    'aflow': ('nomsa', 'aflow', 'original', 'binary'),
    # 'gvpP': ('gvp', 'binary', 'original', 'binary'),
    'gvpL': ('nomsa', 'binary', 'gvp', 'binary'),
    # 'aflow_ring3': ('nomsa', 'aflow_ring3', 'original', 'binary'),
    'gvpL_aflow': ('nomsa', 'aflow', 'gvp', 'binary'),
    # 'gvpL_aflow_rng3': ('nomsa', 'aflow_ring3', 'gvp', 'binary'),
    #GVPL_ESMM_davis3D_nomsaF_aflowE_48B_0.00010636872718329864LR_0.23282479481785903D_2000E_gvpLF_binaryLE
    # 'gvpl_esm_aflow': ('ESM', 'aflow', 'gvp', 'binary'),
}

df = prepare_df('/cluster/home/t122995uhn/projects/MutDTA/results/v113/model_media/model_stats.csv')
fig, axes = fig_combined(df, datasets=['davis'], fig_callable=custom_fig,
             models=models, metrics=['cindex', 'mse'],
             fig_scale=(10,5), add_stats=True, title_postfix=" test set performance")
plt.xticks(rotation=45)

df = prepare_df('/cluster/home/t122995uhn/projects/MutDTA/results/v113/model_media/model_stats_val.csv')
fig, axes = fig_combined(df, datasets=['davis'], fig_callable=custom_fig,
             models=models, metrics=['cindex', 'mse'],
             fig_scale=(10,5), add_stats=True, title_postfix=" validation set performance")
plt.xticks(rotation=45)
