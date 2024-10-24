#%%
import os
print("os.env.TRANSFORMERS_CACHE - ", os.environ.get('TRANSFORMERS_CACHE'))
print("           os.env.HF_HOME - ", os.environ.get('HF_HOME'))
print("    os.env.HF_HUB_OFFLINE - ", os.environ.get('HF_HUB_OFFLINE'))

import torch
from tqdm import tqdm
from src import cfg

from src.utils.loader import Loader
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print("\n############### After imports: #############")
print("os.env.TRANSFORMERS_CACHE - ", os.environ['TRANSFORMERS_CACHE'])
print("           os.env.HF_HOME - ", os.environ['HF_HOME'])
print("    os.env.HF_HUB_OFFLINE - ", os.environ['HF_HUB_OFFLINE'])
m, _ = Loader.load_tuned_model('davis_esm', fold=0, device=DEVICE)
m.eval()


print("done")

exit()





















#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

new = '/cluster/home/t122995uhn/projects/splits/new/pdbbind/'

train_df = pd.concat([pd.read_csv(f'{new}train0.csv'), 
                      pd.read_csv(f'{new}val0.csv')], axis=0)
test_df = pd.read_csv(f'{new}test.csv')

all_df = pd.concat([train_df, test_df], axis=0)
print(len(all_df))


#%%
old = '/cluster/home/t122995uhn/projects/splits/old/pdbbind/'
old_test_df = pd.read_csv(f'{old}test.csv')
old_train_df = all_df[~all_df['code'].isin(old_test_df['code'])]

# %%
# this will give us an estimate to how well targeted the training proteins are vs the test proteins
def proteins_targeted(train_df, test_df, split='new', min_freq=0, normalized=False):
    # protein count comparison (number of diverse proteins)
    plt.figure(figsize=(18,8))
    # x-axis is the normalized frequency, y-axis is the number of proteins that have that frequency (also normalized)
    vc = train_df.prot_id.value_counts()
    vc = vc[vc > min_freq]
    train_counts = list(vc/len(test_df)) if normalized else vc.values
    vc = test_df.prot_id.value_counts()
    vc = vc[vc > min_freq]
    test_counts = list(vc/len(test_df)) if normalized else vc.values

    sns.histplot(train_counts, 
                bins=50, stat='density', color='green', alpha=0.4)
    sns.histplot(test_counts, 
                bins=50,stat='density', color='blue', alpha=0.4)

    sns.kdeplot(train_counts, color='green', alpha=0.8)
    sns.kdeplot(test_counts, color='blue', alpha=0.8)

    plt.xlabel(f"{'normalized ' if normalized else ''} frequency")
    plt.ylabel("normalized number of proteins with that frequency")
    plt.title(f"Targeted differences for {split} split{f' (> {min_freq})' if min_freq else ''}")
    if not normalized:
        plt.xlim(-8,100)

# proteins_targeted(old_train_df, old_test_df, split='oncoKB')
# plt.show()
# proteins_targeted(train_df, test_df, split='random')
# plt.show()


proteins_targeted(old_test_df, test_df, split='oncoKB(green) vs random(blue) test')
plt.show()
proteins_targeted(old_test_df, test_df, split='oncoKB(green) vs random(blue) test', min_freq=5)
plt.show()
proteins_targeted(old_test_df, test_df, split='oncoKB(green) vs random(blue) test', min_freq=10)
plt.show()
proteins_targeted(old_test_df, test_df, split='oncoKB(green) vs random(blue) test', min_freq=15)
plt.show()
proteins_targeted(old_test_df, test_df, split='oncoKB(green) vs random(blue) test', min_freq=20)
plt.show()
# proteins_targeted(old_train_df, train_df, split='oncoKB(green) vs random train')
# plt.show()
#%% sequence length comparison
def seq_kde(all_df, train_df, test_df, split='new'):
    plt.figure(figsize=(12, 8))

    sns.kdeplot(all_df.prot_seq.str.len().reset_index()['prot_seq'], label='All', color='blue')
    sns.kdeplot(train_df.prot_seq.str.len().reset_index()['prot_seq'], label='Train', color='green')
    sns.kdeplot(test_df.prot_seq.str.len().reset_index()['prot_seq'], label='Test', color='red')

    plt.xlabel('Sequence Length')
    plt.ylabel('Density')
    plt.title(f'Sequence Length Distribution ({split} split)')
    plt.legend()

seq_kde(all_df,train_df,test_df, split='new')
plt.show()
seq_kde(all_df,old_train_df,old_test_df, split='old')

# %%
from Bio import pairwise2
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Align import substitution_matrices

from tqdm import tqdm
import random

def get_group_similarity(group1, group2):
    # Choose a substitution matrix (e.g., BLOSUM62)
    matrix = substitution_matrices.load("BLOSUM62")

    # Define gap penalties
    gap_open = -10
    gap_extend = -0.5

    # Function to calculate pairwise similarity score
    def calculate_similarity(seq1, seq2):
        alignments = pairwise2.align.globalds(seq1, seq2, matrix, gap_open, gap_extend)
        return alignments[0][2]  # Return the score of the best alignment

    # Compute pairwise similarity between all sequences in group1 and group2
    similarity_scores = []
    for seq1 in group1:
        for seq2 in group2:
            score = calculate_similarity(seq1, seq2)
            similarity_scores.append(score)

    # Calculate the average similarity score
    average_similarity = sum(similarity_scores) / len(similarity_scores)
    return similarity_scores, average_similarity


# sample 10 sequences randomly 100x
train_seq = old_train_df.prot_seq.drop_duplicates().to_list()
test_seq = old_test_df.prot_seq.drop_duplicates().to_list()
sample_size = 5
trials = 100

est_similarity = 0
for _ in tqdm(range(trials)):
    _, avg = get_group_similarity(random.sample(train_seq, sample_size), 
                                  random.sample(test_seq, sample_size))
    est_similarity += avg

print(est_similarity/1000)




# %%
# building pocket datasets:
from src.utils.pocket_alignment import pocket_dataset_full
import shutil
import os

data_dir = '/cluster/home/t122995uhn/projects/data/'
db_type = ['kiba', 'davis']
db_feat = ['nomsa_binary_original_binary', 'nomsa_aflow_original_binary', 
           'nomsa_binary_gvp_binary',      'nomsa_aflow_gvp_binary']

for t in db_type:
    for f in db_feat:
        print(f'\n---{t}-{f}---\n')
        dataset_dir= f"{data_dir}/DavisKibaDataset/{t}/{f}/full"
        save_dir   = f"{data_dir}/v131/DavisKibaDataset/{t}/{f}/full"
        
        pocket_dataset_full(
            dataset_dir= dataset_dir,
            pocket_dir = f"{data_dir}/{t}/",
            save_dir   = save_dir,
            skip_download=True
        )
        
#%%
import pandas as pd

def get_test_oncokbs(train_df=pd.read_csv('/cluster/home/t122995uhn/projects/data/test/PDBbindDataset/nomsa_binary_original_binary/full/cleaned_XY.csv'),
                     oncokb_fp='/cluster/home/t122995uhn/projects/data/tcga/mart_export.tsv', 
                     biomart='/cluster/home/t122995uhn/projects/downloads/oncoKB_DrugGenePairList.csv'):
    #Get gene names for PDBbind
    dfbm = pd.read_csv(oncokb_fp, sep='\t')
    dfbm['PDB ID'] = dfbm['PDB ID'].str.lower()
    train_df.reset_index(names='idx',inplace=True)

    df_uni = train_df.merge(dfbm, how='inner', left_on='prot_id', right_on='UniProtKB/Swiss-Prot ID')
    df_pdb = train_df.merge(dfbm, how='inner', left_on='code', right_on='PDB ID')

    # identifying ovelap with oncokb
    # df_all will have duplicate entries for entries with multiple gene names...
    df_all = pd.concat([df_uni, df_pdb]).drop_duplicates(['idx', 'Gene name'])[['idx', 'code', 'Gene name']]

    dfkb = pd.read_csv(biomart)
    df_all_kb = df_all.merge(dfkb.drop_duplicates('gene'), left_on='Gene name', right_on='gene', how='inner')

    trained_genes = set(df_all_kb.gene)

    #Identify non-trained genes
    return dfkb[~dfkb['gene'].isin(trained_genes)], dfkb[dfkb['gene'].isin(trained_genes)], dfkb


train_df = pd.read_csv('/cluster/home/t122995uhn/projects/data/test/PDBbindDataset/nomsa_binary_original_binary/train0/cleaned_XY.csv')
val_df = pd.read_csv('/cluster/home/t122995uhn/projects/data/test/PDBbindDataset/nomsa_binary_original_binary/val0/cleaned_XY.csv')

train_df = pd.concat([train_df, val_df])

get_test_oncokbs(train_df=train_df)

#%%
##############################################################################
########################## BUILD/SPLIT DATASETS ##############################
##############################################################################
import os
from src.data_prep.init_dataset import create_datasets
from src import cfg
import logging
cfg.logger.setLevel(logging.DEBUG)

dbs = [cfg.DATA_OPT.davis, cfg.DATA_OPT.kiba]
splits = ['davis', 'kiba']
splits = ['/cluster/home/t122995uhn/projects/MutDTA/splits/' + s for s in splits]
print(splits)

#%%
for split, db in zip(splits, dbs):
    print('\n',split, db)
    create_datasets(db, 
                feat_opt=cfg.PRO_FEAT_OPT.nomsa, 
                edge_opt=[cfg.PRO_EDGE_OPT.binary, cfg.PRO_EDGE_OPT.aflow],
                ligand_features=[cfg.LIG_FEAT_OPT.original, cfg.LIG_FEAT_OPT.gvp], 
                ligand_edges=cfg.LIG_EDGE_OPT.binary, overwrite=False,
                k_folds=5,

                test_prots_csv=f'{split}/test.csv',
                val_prots_csv=[f'{split}/val{i}.csv' for i in range(5)])

#%% TEST INFERENCE
from src import cfg
from src.utils.loader import Loader

# db2 = Loader.load_dataset(cfg.DATA_OPT.davis, 
#                          cfg.PRO_FEAT_OPT.nomsa, cfg.PRO_EDGE_OPT.aflow,
#                          path='/cluster/home/t122995uhn/projects/data/',
#                          subset="full")

db2 = Loader.load_DataLoaders(cfg.DATA_OPT.davis, 
                         cfg.PRO_FEAT_OPT.nomsa, cfg.PRO_EDGE_OPT.aflow,
                         path='/cluster/home/t122995uhn/projects/data/v131',
                         training_fold=0,
                         batch_train=2)
for b2 in db2['test']: break


# %%
m = Loader.init_model(cfg.MODEL_OPT.DG, cfg.PRO_FEAT_OPT.nomsa, cfg.PRO_EDGE_OPT.aflow,
                  dropout=0.3480, output_dim=256,
                  )

#%%
# m(b['protein'], b['ligand'])
m(b2['protein'], b2['ligand'])
#%%
model = m
loaders = db2
device = 'cpu'
NUM_EPOCHS = 1
LEARNING_RATE = 0.001
from src.train_test.training import train

logs = train(model, loaders['train'], loaders['val'], device, 
            epochs=NUM_EPOCHS, lr_0=LEARNING_RATE)
