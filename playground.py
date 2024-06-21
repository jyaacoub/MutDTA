# %%
import pandas as pd
from src.data_prep.downloaders import Downloader

df = pd.read_csv('../data/all_prots.csv')

id_status = {}
for db in df.db.unique():
    id = Downloader.download_pocket_seq(df[df.db == db].prot_id.to_list(), 
                                        f"../data/pocket_seq/{db}/",
                                        tqdm_desc=f"Downloading {db} pocket sequences")
    id_status[db] = id
#%%
import json
# json.dump(id_status, open('../data/pocket_seq/seq_out.json', 'w'))
# id_status = json.load(open('../data/pocket_seq/seq_out.json', 'r'))
for db, st in id_status.items():
    total_ids = len(st)
    missing = list(id_status[db].values()).count(400)
    print(f"{db}: {total_ids - missing}/{total_ids} ({missing})")

# %%
########################################################################
########################## TEST DATASETS ###############################
########################################################################
from src import config as cfg
from src.utils.loader import Loader


# load up platinum test db
loaders = Loader.load_DataLoaders(cfg.DATA_OPT.platinum,
                            pro_feature    = cfg.PRO_FEAT_OPT.nomsa, 
                            edge_opt       = cfg.PRO_EDGE_OPT.binary,
                            ligand_feature = cfg.LIG_FEAT_OPT.original, 
                            ligand_edge    = cfg.LIG_EDGE_OPT.binary,
                            datasets=['test'])


# %%
########################################################################
####################### VIOLIN PLOTS ###################################
########################################################################
import logging
from typing import OrderedDict

import seaborn as sns
from matplotlib import pyplot as plt
from statannotations.Annotator import Annotator

from src.analysis.figures import prepare_df, custom_fig, fig_combined

df = prepare_df()
# %%
models = {
    'DG': ('nomsa', 'binary', 'original', 'binary'),
    'aflow': ('nomsa', 'aflow', 'original', 'binary'),
    # 'aflow_ring3': ('nomsa', 'aflow_ring3', 'original', 'binary'),
    # 'gvpP': ('gvp', 'binary', 'original', 'binary'),
    # 'gvpL': ('nomsa', 'binary', 'gvp', 'binary'),
    'gvpL_aflow': ('nomsa', 'aflow', 'gvp', 'binary'),
    'gvpL': ('nomsa', 'binary', 'gvp', 'binary'),
    # 'gvpL_aflow_rng3': ('nomsa', 'aflow_ring3', 'gvp', 'binary'),
}

# %%
fig, axes = fig_combined(df, datasets=['davis','PDBbind'], fig_callable=custom_fig,
             models=models, metrics=['cindex', 'mse'],
             fig_scale=(8,5))
plt.xticks(rotation=45)


# %%
########################################################################
########################## PLATINUM ANALYSIS ###########################
########################################################################
import torch, os
import pandas as pd

from src import cfg
from src import TUNED_MODEL_CONFIGS
from src.utils.loader import Loader
from src.train_test.training import test
from src.analysis.figures import predictive_performance, tbl_stratified_dpkd_metrics, tbl_dpkd_metrics_overlap, tbl_dpkd_metrics_in_binding

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

INFERENCE = True
VERBOSE = True
out_dir = f'{cfg.MEDIA_SAVE_DIR}/test_set_pred/'
os.makedirs(out_dir, exist_ok=True)
cp_dir = cfg.CHECKPOINT_SAVE_DIR
RAW_PLT_CSV=f"{cfg.DATA_ROOT}/PlatinumDataset/raw/platinum_flat_file.csv"

#%% load up model:
for KEY, CONFIG in TUNED_MODEL_CONFIGS.items():
    MODEL_KEY = lambda fold: Loader.get_model_key(CONFIG['model'], CONFIG['dataset'], CONFIG['feature_opt'], CONFIG['edge_opt'], 
                                    CONFIG['batch_size'], CONFIG['lr'], CONFIG['architecture_kwargs']['dropout'],
                                    n_epochs=2000, fold=fold, 
                                    ligand_feature=CONFIG['lig_feat_opt'], ligand_edge=CONFIG['lig_edge_opt'])
    print('\n\n'+ '## ' + KEY)
    OUT_PLT = lambda i: f'{out_dir}/{MODEL_KEY(i)}_PLATINUM.csv'
    db_p = f"{CONFIG['feature_opt']}_{CONFIG['edge_opt']}_{CONFIG['lig_feat_opt']}_{CONFIG['lig_edge_opt']}"
    
    if CONFIG['dataset'] in ['kiba', 'davis']:
        db_p = f"DavisKibaDataset/{CONFIG['dataset']}/{db_p}"
    else:
        db_p = f"{CONFIG['dataset']}Dataset/{db_p}"
        
    train_p = lambda set: f"{cfg.DATA_ROOT}/{db_p}/{set}0/cleaned_XY.csv"
    
    if not os.path.exists(OUT_PLT(0)) and INFERENCE:
        print('running inference!')
        cp = lambda fold: f"{cp_dir}/{MODEL_KEY(fold)}.model"
        
        model = Loader.init_model(model=CONFIG["model"], pro_feature=CONFIG["feature_opt"],
                                    pro_edge=CONFIG["edge_opt"],**CONFIG['architecture_kwargs'])

        # load up platinum test db
        loaders = Loader.load_DataLoaders(cfg.DATA_OPT.platinum,
                                    pro_feature    = CONFIG['feature_opt'], 
                                    edge_opt       = CONFIG['edge_opt'],
                                    ligand_feature = CONFIG['lig_feat_opt'], 
                                    ligand_edge    = CONFIG['lig_edge_opt'],
                                    datasets=['test'])

        for i in range(5):
            model.safe_load_state_dict(torch.load(cp(i), map_location=device))
            model.to(device)
            model.eval()

            loss, pred, actual = test(model, loaders['test'], device, verbose=True)
            
            # saving as csv with columns code, pred, actual
            # get codes from test loader
            codes, pid = [b['code'][0] for b in loaders['test']], [b['prot_id'][0] for b in loaders['test']]
            df = pd.DataFrame({'prot_id': pid, 'pred': pred, 'actual': actual}, index=codes)
            df.index.name = 'code'
            df.to_csv(OUT_PLT(i))

    # run platinum eval:
    print('\n### 1. predictive performance')
    mkdown = predictive_performance(OUT_PLT, train_p, verbose=VERBOSE, plot=False)
    print('\n### 2 Mutation impact analysis')
    print('\n#### 2.1 $\Delta pkd$ predictive performance')
    mkdn = tbl_dpkd_metrics_overlap(OUT_PLT, train_p, verbose=VERBOSE, plot=False)
    print('\n#### 2.2 Stratified by location of mutation (binding pocket vs not in binding pocket)')
    m = tbl_dpkd_metrics_in_binding(OUT_PLT, RAW_PLT_CSV, verbose=VERBOSE, plot=False)
    
# %%
dfr = pd.read_csv(RAW_PLT_CSV, index_col=0)

# add in_binding info to df
def get_in_binding(df, dfr):
    """
    df is the predicted csv with index as <raw_idx>_wt (or *_mt) where raw_idx 
    corresponds to an index in dfr which contains the raw data for platinum including 
    ('mut.in_binding_site')
        - 0: wildtype rows
        - 1: close (<8 Ang)
        - 2: Far (>8 Ang)
    """
    pocket = dfr[dfr['mut.in_binding_site'] == 'YES'].index   
    pclass = []
    for code in df.index:
        if '_wt' in code:
            pclass.append(0)
        elif int(code.split('_')[0]) in pocket:
            pclass.append(1)
        else:
            pclass.append(2)
    df['pocket'] = pclass
    return df

df = get_in_binding(pd.read_csv(OUT_PLT(0), index_col=0), dfr)
if VERBOSE: 
    cnts = df.pocket.value_counts()
    cnts.index = ['wt', 'in pocket', 'not in pocket']
    cnts.name = "counts"
    print(cnts.to_markdown(), end="\n\n")

tbl_stratified_dpkd_metrics(OUT_PLT, NORMALIZE=True, n_models=5, df_transform=get_in_binding,
                            conditions=['(pocket == 0) | (pocket == 1)', '(pocket == 0) | (pocket == 2)'], 
                            names=['in pocket', 'not in pocket'], 
                            verbose=VERBOSE, plot=True, dfr=dfr)

