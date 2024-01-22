# %% Checking overlap of proteins across davis, kiba and PDBbind
import pandas as pd

pdbbind_csv_p = "/cluster/home/t122995uhn/projects/data/PDBbindDataset/nomsa_binary_original_binary/full/XY.csv"
davis_csv_p =   "/cluster/home/t122995uhn/projects/data/DavisKibaDataset/davis/nomsa_binary_original_binary/full/XY.csv"
kiba_csv_p =    "/cluster/home/t122995uhn/projects/data/DavisKibaDataset/kiba/nomsa_binary_original_binary/full/XY.csv"



pdbbind_df = pd.read_csv(pdbbind_csv_p)
davis_df = pd.read_csv(davis_csv_p)
kiba_df = pd.read_csv(kiba_csv_p)

#%%
# dropping duplicates across prot_id and prot_seq independently
pdbbind_uni_id = pdbbind_df.prot_id.drop_duplicates(keep='first')
davis_uni_id = davis_df.prot_id.drop_duplicates(keep='first')
kiba_uni_id = kiba_df.prot_id.drop_duplicates(keep='first')

pdbbind_uni_seq = pdbbind_df.prot_seq.drop_duplicates(keep='first')
davis_uni_seq = davis_df.prot_seq.drop_duplicates(keep='first')
kiba_uni_seq = kiba_df.prot_seq.drop_duplicates(keep='first')

# %% keep only those indexes that show up in both unique id and unique seq.
pdbbind_uni = pd.merge(pdbbind_uni_id, pdbbind_uni_seq, left_index=True, right_index=True, how='inner')
davis_uni = pd.merge(davis_uni_id, davis_uni_seq, left_index=True, right_index=True, how='inner')
kiba_uni = pd.merge(kiba_uni_id, kiba_uni_seq, left_index=True, right_index=True, how='inner')

pdbbind_uni_seq = pdbbind_df.iloc[pdbbind_uni.index]['prot_seq']
davis_uni_seq = davis_df.iloc[davis_uni.index]['prot_seq']
kiba_uni_seq = kiba_df.iloc[kiba_uni.index]['prot_seq']


print("Individual =",len(pdbbind_uni_seq), len(davis_uni_seq), len(kiba_uni_seq))

all_uni_seq = pd.concat([pdbbind_uni_seq, davis_uni_seq, kiba_uni_seq])
all_uni_seq.reset_index(drop=True, inplace=True)

print("Total concat =", len(all_uni_seq))
print('Total unique across all 3 =', len(all_uni_seq.unique()))

unique_pdbbind_davis = pd.concat([pdbbind_uni_seq, davis_uni_seq]).unique()
unique_pdbbind_kiba = pd.concat([pdbbind_uni_seq, kiba_uni_seq]).unique()
unique_kiba_davis = pd.concat([kiba_uni_seq, davis_uni_seq]).unique()

print('Total unique between pdbbind and davis =', len(unique_pdbbind_davis))
print('Total unique between pdbbind and kiba =', len(unique_pdbbind_kiba))
print('Total unique between kiba and davis =', len(unique_kiba_davis))
print('')

print('Shared between pdbbind and davis =', (len(pdbbind_uni) + len(davis_uni)) - len(unique_pdbbind_davis))
print('Shared between pdbbind and kiba =', (len(pdbbind_uni) + len(kiba_uni)) - len(unique_pdbbind_kiba))
print('Shared between kiba and davis =', (len(kiba_uni) + len(davis_uni)) - len(unique_kiba_davis))


# %%
# import torch

# from src.utils.loader import Loader
# from src.train_test.utils import debug
# from src.utils import config as cfg
# from torch_geometric.utils import dropout_edge, dropout_node

# device = torch.device('cuda:0')#'cuda:0' if torch.cuda.is_available() else 'cpu')

# MODEL, DATA,  = 'SPD', 'davis'
# FEATURE, EDGEW = 'foldseek', 'binary'
# ligand_feature, ligand_edge = None, None
# BATCH_SIZE = 20
# fold = 0
# pro_overlap = False
# DROPOUT = 0.2


# # ==== LOAD DATA ====
# loaders = Loader.load_DataLoaders(data=DATA, pro_feature=FEATURE, edge_opt=EDGEW, path=cfg.DATA_ROOT, 
#                                     ligand_feature=ligand_feature, ligand_edge=ligand_edge,
#                                     batch_train=BATCH_SIZE,
#                                     datasets=['train'],
#                                     training_fold=fold,
#                                     protein_overlap=pro_overlap)


# # ==== LOAD MODEL ====
# print(f'#Device: {device}')
# model = Loader.init_model(model=MODEL, pro_feature=FEATURE, pro_edge=EDGEW, dropout=DROPOUT,
#                             ligand_feature=ligand_feature, ligand_edge=ligand_edge).to(device)


# # %%
# train, eval = debug(model, loaders['train'], device=device)
# # %%

# %%
