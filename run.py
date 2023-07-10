#%%
import os, random, itertools, math

from tqdm import tqdm
import pandas as pd
import numpy as np
import torch

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from src.models.prior_work import DGraphDTA
from src.data_processing import PDBbindDataset, train_val_test_split
from src.models import train, test
from src.data_analysis import get_metrics


PDB_RAW_DIR = '../data/v2020-other-PL/'
PDB_PROCESSED_DIR = '../data//PDBbindDataset/msa/' #NOTE: type of dataset specified here 
ALN_DIR = '../data/msa/outputs/'
MODEL_STATS_CSV = 'results/model_media/model_stats.csv'
#loading data and splitting into train, val, test
pdb_dataset = PDBbindDataset(PDB_PROCESSED_DIR, PDB_RAW_DIR, ALN_DIR,
                             cmap_threshold=8.0,
                             shannon=False)

# Dataset Hyperparameters
TRAIN_SPLIT= .8 # 80% of data for training
VAL_SPLIT = .1 # 10% for val and remaining is for testing (10%)
SHUFFLE_DATA = True
RAND_SEED=0

random.seed(RAND_SEED)
np.random.seed(RAND_SEED)
torch.manual_seed(RAND_SEED)

# Tune Hyperparameters after grid search
BATCH_SIZE = 64
LEARNING_RATE = 0.001
DROPOUT = 0.2
NUM_EPOCHS = 50
weight_opt = ['kiba', 'davis', 'random']
WEIGHTS = 'davis'

SAVE_RESULTS = True
media_save_p = 'results/model_media/figures/'

#%% training and testing
model_key = lambda x: f'{x}W_{NUM_EPOCHS}E_msa'
metrics = {}
for WEIGHTS in weight_opt:
    MODEL_KEY = model_key(WEIGHTS)
    # {BATCH_SIZE}B_{LEARNING_RATE}LR_{DROPOUT}DO are fixed so not included in model key
    mdl_save_p = f'results/model_checkpoints/ours/DGraphDTA_{MODEL_KEY}.model'

    train_loader, val_loader, test_loader = train_val_test_split(pdb_dataset, 
                        train_split=TRAIN_SPLIT, val_split=VAL_SPLIT,
                        shuffle_dataset=True, random_seed=RAND_SEED, 
                        batch_size=BATCH_SIZE, use_refined=True)

    # loading model:
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    print(f'\n{MODEL_KEY}')
        
    model = DGraphDTA(dropout=DROPOUT)
    model.to(device)
    assert WEIGHTS in weight_opt, 'WEIGHTS must be one of: kiba, davis, random'
    if WEIGHTS != 'random':
        model_file_name = f'results/model_checkpoints/prior_work/DGraphDTA_{WEIGHTS}_t2.model'
        model.load_state_dict(torch.load(model_file_name, map_location=device))

    # training
    logs = train(model, train_loader, val_loader, device, 
            epochs=NUM_EPOCHS, lr=LEARNING_RATE)
    # saving model checkpoint
    torch.save(model.state_dict(), mdl_save_p)
    print(f'Model saved to: {mdl_save_p}')

    ax = plt.figure().gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    index = np.arange(1, NUM_EPOCHS+1)
    plt.plot(index, logs['train_loss'], label='train')
    plt.plot(index, logs['val_loss'], label='val')
    plt.legend()
    plt.title(f'{MODEL_KEY} Loss')
    plt.xlabel('Epoch')
    # plt.xticks(range(0,NUM_EPOCHS+1, 2))
    plt.xlim(0, NUM_EPOCHS)
    plt.ylabel('Loss')
    if SAVE_RESULTS: plt.savefig(f'results/model_media/figures/{MODEL_KEY}_loss.png')
    plt.show()
    
    #  testing
    loss, pred, actual = test(model, test_loader, device)
    get_metrics(pred, actual,
                save_results=SAVE_RESULTS,
                save_path=media_save_p,
                model_key=MODEL_KEY,
                csv_file=MODEL_STATS_CSV
                )
    metrics[MODEL_KEY] = {'test_loss': loss,
                          'logs': logs}

# %%
#plotting all model train and val loss together
index = np.arange(1, NUM_EPOCHS+1)
for w in weight_opt:
    c = plt.plot(index, metrics[model_key(w)]['logs']['train_loss'], 
                 label=f'{w} train')[0].get_color()
    plt.plot(index, metrics[model_key(w)]['logs']['val_loss'], 
             label=f'{w} val', color=c, alpha=0.5)

plt.legend()
plt.title('DGraphDTA Weight Initialization Impact on Loss')
plt.xlabel('Epoch')
plt.xlim(0, NUM_EPOCHS)
plt.ylabel('Loss')
plt.savefig(f'results/model_media/figures/all_loss.png')
plt.show()


# %%
test_codes = np.array([])
test_labels = np.array([])
for p,l in test_loader:
    test_codes= np.append(test_codes, p.code)
    test_labels = np.append(test_labels, p.y.numpy().flatten())
    
test_df = pd.DataFrame(test_labels, index=test_codes, columns=['pkd'])
test_df.index.name = 'PDBCode'

vina_df = pd.read_csv('results/PDBbind/vina_out/run10.csv', index_col=0)
common = vina_df.merge(test_df, on='PDBCode')

# Test set is chosen to be all from refined set to 100% match vina
print(len(common), 'codes in common with vina out of', len(test_df), 'test codes')

# %% Calc results on this common test set
log_y = common['pkd']
R=0.0019870937 # kcal/Mol*K (gas constant)
T=273.15       # K
RT = R*T
log_z = -np.log(math.e**(common['vina_deltaG(kcal/mol)']/RT)) # TODO: check if this is correct

get_metrics(log_y, log_z,
            save_results=SAVE_RESULTS,
            save_path=media_save_p,
            model_key='10_vina_common',
            csv_file=MODEL_STATS_CSV,
            show=True)


# %% displaying models results
# creating the dataset loader:

for mkey in metrics:
    model = DGraphDTA()
    model_file_name = f'results/model_checkpoints/ours/DGraphDTA_{mkey}.model'
    print(f'\n\n{mkey}')
    model.load_state_dict(torch.load(model_file_name, map_location=device))
    model.eval()
    
    log_z = np.array([])
    log_y = np.array([])
    for prots, ligs in test_loader:
        pred = model(prots, ligs).detach().cpu().numpy().flatten()
        log_z = np.append(log_z, pred)
        log_y = np.append(log_y, prots.y.detach().cpu().numpy().flatten())
                
    get_metrics(log_y, log_z,
                save_results=SAVE_RESULTS,
                save_path=media_save_p,
                model_key=mkey,
                csv_file=MODEL_STATS_CSV,
                show=True)
            

# %%
#%% plot bar graph of results by row (comparing vina to DGraphDTA)
# cols are: run,cindex,pearson,spearman,mse,mae,rmse
df_res = pd.read_csv(MODEL_STATS_CSV)[-4:]

for col in df_res.columns[1:]:
    plt.figure()
    bars = plt.bar(df_res['run'],df_res[col])
    bars[0].set_color('red')
    bars[1].set_color('green')
    bars[-1].set_color('black')
    plt.title(col)
    plt.xlabel('run')
    plt.xticks(rotation=30)
    # plt.ylim((0.2, 0.8))
    plt.show()
    
# %% plot bar graph (comparing to no msa)
df_res = pd.read_csv(MODEL_STATS_CSV, index_col=0)
msa = df_res[df_res.index.str.contains('_msa')]
no_msa = df_res.loc[[i[:-4] for i in msa.index]]
both = msa.add(no_msa, fill_value=0)
for col in both.columns:
    plt.figure()
    for w in weight_opt:
        k = model_key(w)
        c = plt.bar([w], both.loc[k,col], label=w)[0]._original_facecolor
        k= k[:-4]
        plt.bar([f'{w}_msa'], both.loc[k,col], color=c, alpha=0.6)
    
    plt.title(f'{col} with and without MSA')
    # for b in bars:
    # plt.legend()
    plt.xticks(rotation=30)
    plt.savefig(f'results/model_media/figures/{col}_msacompare.png')
    plt.show()

# %%
