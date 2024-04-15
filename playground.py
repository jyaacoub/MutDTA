# %% Create and test on platinum dataset:
from src.data_prep.init_dataset import create_datasets
from src import config as cfg

create_datasets(cfg.DATA_OPT.platinum,
                cfg.PRO_FEAT_OPT.nomsa,
                cfg.PRO_EDGE_OPT.binary,
                ligand_features=cfg.LIG_FEAT_OPT.original,
                ligand_edges=cfg.LIG_EDGE_OPT.binary,
                k_folds=None,
# just a test dataset, no training can be done on this
                train_split=0.0, val_split=0.0) 

#%%
import pandas as pd
data_p = "/home/jean/projects/data/PlatinumDataset"
csv = f"{data_p}/nomsa_binary_original_binary/full/cleaned_XY.csv"
csv_raw = f"{data_p}/nomsa_binary_original_binary/full/XY.csv"

df = pd.read_csv(csv, index_col=0)
df_raw = pd.read_csv(csv_raw, index_col=0)

# sorting by sequence length to keep the longest protein sequence 
# instead of just the first.
if 'sort_order' in df:
    # ensures that the right codes are always present in unique_pro
    df.sort_values(by='sort_order', inplace=True)
else:
    df['seq_len'] = df['prot_seq'].str.len()
    df.sort_values(by='seq_len', ascending=False, inplace=True)
    df['sort_order'] = [i for i in range(len(df))]

# Get unique protid codes
idx_name = df.index.name
df.reset_index(drop=False, inplace=True)
unique_pro = df[['prot_id']].drop_duplicates(keep='first')

# reverting index to code-based index
df.set_index(idx_name, inplace=True)
unique_df = df.iloc[unique_pro.index]



# %%
from src.data_prep.datasets import PlatinumDataset

PlatinumDataset.get_unique_prots(df)

# %%
