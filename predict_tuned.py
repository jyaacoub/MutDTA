#%%
########################################################################
###################### TEST MODEL W/DATASET ############################
########################################################################
import os
import torch
from tqdm import tqdm
import pandas as pd

from src import config as cfg
from src import TUNED_MODEL_CONFIGS
from src.utils.loader import Loader
from src.models.utils import BaseModel
from src.data_prep.init_dataset import create_datasets

def get_check_p(params, fold=0):
    key = Loader.get_model_key(params['model'], params['dataset'], params['feature_opt'], 
                            params['edge_opt'], params['batch_size'], params['lr'], fold=fold, 
                            ligand_feature=params['lig_feat_opt'], ligand_edge=params['lig_edge_opt'], 
                            **params['architecture_kwargs'])
    model_p_tmp = f'{cfg.MODEL_SAVE_DIR}/{key}.model_tmp'
    model_p = f'{cfg.MODEL_SAVE_DIR}/{key}.model'

    # MODEL_KEY = 'DDP-' + MODEL_KEY # distributed model
    model_p = model_p if os.path.isfile(model_p) else model_p_tmp
    assert os.path.isfile(model_p), f"MISSING MODEL CHECKPOINT {model_p}"
    return key, model_p

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
targets = ['davis_gvpl', 'davis_DG']    # ['davis_gvpl_aflow', 'davis_gvpl', 'davis_aflow', 'davis_DG']
for t in targets:
    for fold in range(5):
        print(f'\n{"-"*20} {t}-{fold} {"-"*20}')
        params = TUNED_MODEL_CONFIGS[t]

        m : BaseModel = Loader.init_model(params['model'],params['feature_opt'], params['edge_opt'], **params['architecture_kwargs'])
        key, model_p = get_check_p(params, fold=fold)
        
        if os.path.isfile(f'/cluster/home/t122995uhn/projects/data/predictions/{key}.csv'):
            print("Already predicted!")
            continue
        
        print('\t', model_p)
        m.to(device)
        sd = torch.load(model_p, map_location=device)
        try:
            m.safe_load_state_dict(sd)
        except Exception as e:
            if 'dense_out.9.bias' in sd:
                print(e)
                sd['dense_out.8.weight'] = sd['dense_out.9.weight']
                sd['dense_out.8.bias'] = sd['dense_out.9.bias']
                del sd['dense_out.9.bias']
                del sd['dense_out.9.weight']
                
                m.safe_load_state_dict(sd)
                torch.save(sd, model_p)
            else:
                raise e

        kwargs = {
            "ligand_features": params['lig_feat_opt'],
            "ligand_edges": params['lig_edge_opt'],
            "k_folds": 5, 
            "test_prots_csv": '/cluster/home/t122995uhn/projects/downloads/test_prots_gene_names.csv'
        }

        kwargs = {k: v for k, v in kwargs.items() if v is not None}

        create_datasets(params['dataset'], params['feature_opt'], 
                        params['edge_opt'],**kwargs)
        
        try:
            dl = Loader.load_DataLoaders(params['dataset'], params['feature_opt'], 
                                        params['edge_opt'], 
                                        ligand_feature=params['lig_feat_opt'], 
                                        ligand_edge=params['lig_edge_opt'], 
                                        datasets=['test'])['test']
            sample = next(iter(dl))
        except Exception:
            print('FAILED TO LOAD DATA FOR', t)
            continue

        df_dict = {
            "pred": [],
            "actual": [],
            "prot_id": [],
            'lig_seq': [],
            "seq_len": [],
        }
        m.eval()
        for d in tqdm(dl):
            df_dict['pred'].append(m(d['protein'], d['ligand']).item())
            df_dict['actual'].append(d['y'].item())
            df_dict['prot_id'].append(d['prot_id'][0])
            
            df_dict['lig_seq'].append(d['ligand'].lig_seq[0] if 'lig_seq' in d['ligand'] else None)
            df_dict['seq_len'].append(len(d['protein'].pro_seq[0]))
            
        # build df and output to csv
        df = pd.DataFrame.from_dict(df_dict, orient='columns')
        df.to_csv(f'/cluster/home/t122995uhn/projects/data/predictions/{key}.csv')
print('DONE!')
exit()

#############################################
##### Plot sequence length vs MSE ###########
#############################################
import os
import torch
from tqdm import tqdm
import pandas as pd
import numpy as np


from src import config as cfg
from src import TUNED_MODEL_CONFIGS
from src.utils.loader import Loader
from src.models.utils import BaseModel
from src.data_prep.init_dataset import create_datasets
import matplotlib.pyplot as plt

def get_check_p(params, fold=0):
    key = Loader.get_model_key(params['model'], params['dataset'], params['feature_opt'], 
                            params['edge_opt'], params['batch_size'], params['lr'], fold=fold, 
                            ligand_feature=params['lig_feat_opt'], ligand_edge=params['lig_edge_opt'], 
                            **params['architecture_kwargs'])
    model_p_tmp = f'{cfg.MODEL_SAVE_DIR}/{key}.model_tmp'
    model_p = f'{cfg.MODEL_SAVE_DIR}/{key}.model'

    # MODEL_KEY = 'DDP-' + MODEL_KEY # distributed model
    model_p = model_p if os.path.isfile(model_p) else model_p_tmp
    assert os.path.isfile(model_p), f"MISSING MODEL CHECKPOINT {model_p}"
    return key, model_p

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
targets = ['davis_gvpl_aflow', 'davis_gvpl', 'davis_aflow', 'davis_DG']

for t in targets:
    mse_folds = []
    for fold in range(5):
        params = TUNED_MODEL_CONFIGS[t]
        key, model_p = get_check_p(params, fold=fold)
        df = pd.read_csv(f'/cluster/home/t122995uhn/projects/data/predictions/{key}.csv')
        mse = (df['pred'] - df['actual'])**2
        mse_folds.append(mse)
    mse_folds = np.array(mse_folds)
    prot_len = df['seq_len'].to_numpy()

    mse_data = []
    for i in range(len(prot_len)):
        mse_data.append([prot_len[i], mse_folds[:, i]])

    # Create a DataFrame for the collected data
    mse_df = pd.DataFrame(mse_data, columns=['seq_len', 'mse_values'])

    # Calculate mean and standard deviation of MSE for each sequence length
    mse_df['mse_mean'] = mse_df['mse_values'].apply(np.mean)
    mse_df['mse_std'] = mse_df['mse_values'].apply(np.std)

    # Plotting
    plt.figure(figsize=(10, 6))
    sc = plt.scatter(mse_df['seq_len'], mse_df['mse_mean'], c=mse_df['mse_std'], 
                     cmap='viridis', s=50, alpha=0.6)
    plt.colorbar(sc, label='Standard Deviation of MSE')
    plt.xlabel('Sequence Length')
    plt.ylabel('Mean MSE')
    plt.title(f'{t} - Sequence Length vs Mean MSE')
    plt.grid(True)
    plt.show()
    