# %%
from src.data_processing.init_dataset import create_datasets

create_datasets(data_opt=['PDBbind'],
            feat_opt=['msa', 'shannon'],
            edge_opt=['binary'], # for anm and af2 we need structures! (see colabfold-highQ)
            pro_overlap=False, k_folds=5, 
            data_root='/cluster/home/t122995uhn/projects/data/')

# %%
import os
from tqdm import tqdm
from src.data_processing.process_msa import MSARunner

pdb_dir = '/cluster/home/t122995uhn/projects/data/PDBbind_aln/'

for f in tqdm(os.listdir(pdb_dir)):
    if f.split('.')[-1] != 'aln': continue
    fp = os.path.join(pdb_dir, f)
    try:
        MSARunner.clean_msa(f_in=fp, f_out=fp)
    except Exception as e:
        raise Exception(f"error on {fp}") from e
# %%
