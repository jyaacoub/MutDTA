#%%
import random
from typing import Callable, Dict, List
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

#%% FIG 1 - TABLE FOR DATASET COUNTS 
def get_USED_dataset_counts(SPLITS_CSVS="./splits/"):
    """Due to memory limitations a couple records were excluded from our runs this is the full count that were actually used"""
    def get_dataset_info(dataset_name):
        csvp=f"{SPLITS_CSVS}/{dataset_name}"
        df = pd.concat([
            pd.read_csv(f"{csvp}/test.csv", index_col=0),
            pd.read_csv(f"{csvp}/train0.csv", index_col=0),
            pd.read_csv(f"{csvp}/val0.csv", index_col=0)
        ])
        n_prots = len(df.prot_id.unique())
        n_ligs = len(df.lig_id.unique())
        total_count = len(df)
        return {"Dataset": dataset_name, "Protein": n_prots, "Compound": n_ligs, "Total Binding Entities": total_count}

# Collect data for each dataset
    datasets = ['davis', 'kiba', 'pdbbind']
    data = [get_dataset_info(dataset) for dataset in datasets]

    # Create a DataFrame

    df = pd.DataFrame(data)

    # Convert DataFrame to Markdown format
    return df.to_markdown(index=False), df

def get_FULL_dataset_counts():
    """This is the actual counts from source locations"""
    markdown_table = f"""
    | Dataset   |   Protein |   Compound |  Total Binding Entities |
    |-----------|-----------|------------|-------------------------|
    | davis     |       442 |         68 |                   30056 |
    | kiba      |       229 |       2111 |                  118254 |
    | pdbbind   |      3889 |      12639 |                   19443 |
    """
    return markdown_table

#%% FIG 1 - SEQUENCE LENGTH DISTRIBUTION
import seaborn as sns
def sequence_length_distributions(SPLITS_CSVS="./splits", dataset_names=['davis', 'kiba', 'pdbbind'],
                                          figsize=(15, 15), bins=20, bw_adjust=1.5):
    """Distribution of sequences for multiple datasets as subplots"""
    fig, axes = plt.subplots(len(dataset_names), 1, figsize=figsize, sharex=True)

    for i, dataset_name in enumerate(dataset_names):
        csvp = f"{SPLITS_CSVS}/{dataset_name}"
        df = pd.concat([
            pd.read_csv(f"{csvp}/test.csv", index_col=0),
            pd.read_csv(f"{csvp}/train0.csv", index_col=0),
            pd.read_csv(f"{csvp}/val0.csv", index_col=0)
        ])
        df['len'] = df.prot_seq.str.len()
        
        sns.histplot(df['len'], bins=bins, alpha=0.5, label=dataset_name, kde=True, 
                     kde_kws={"bw_adjust": bw_adjust}, ax=axes[i])
        
        axes[i].set_xlabel('Protein Sequence length')
        axes[i].set_ylabel('Frequency')
        axes[i].set_title(f'{dataset_name.capitalize()} Dataset')
        axes[i].legend()

    plt.tight_layout()

def overlay_normalized_sequence_length_distribution(SPLITS_CSVS="./splits", dataset_names=['davis', 'kiba', 'pdbbind'],
                                                    figsize=(15, 5), bins=20, bw_adjust=1.5):
    """Overlay normalized distribution of sequences for multiple datasets on the same plot."""
    plt.figure(figsize=figsize)
    colors = ['blue', 'green', 'orange']  # Different colors for datasets

    for dataset_name, color in zip(dataset_names, colors):
        csvp = f"{SPLITS_CSVS}/{dataset_name}"
        try:
            # Combine test, train, and val CSVs
            df = pd.concat([
                pd.read_csv(f"{csvp}/test.csv", index_col=0),
                pd.read_csv(f"{csvp}/train0.csv", index_col=0),
                pd.read_csv(f"{csvp}/val0.csv", index_col=0)
            ])
            df['len'] = df.prot_seq.str.len()

            # Normalize the histogram frequencies by setting `stat="density"`
            sns.histplot(df['len'], bins=bins, kde=True, kde_kws={"bw_adjust": bw_adjust},
                         alpha=0.5, label=dataset_name.capitalize(), color=color, stat='density')
        except FileNotFoundError:
            print(f"Files for dataset '{dataset_name}' not found. Skipping.")

    # Set labels and title
    plt.xlabel('Protein Sequence Length')
    plt.ylabel('Normalized Frequency (Density)')
    plt.title('Overlayed Normalized Histogram of Protein Sequence Lengths')
    plt.legend()

#%% FIG 1 - MODEL RESULTS
#########################
from matplotlib import pyplot as plt
from src.analysis.figures import prepare_df, fig_combined, custom_fig

def plot_model_results(stats_csv="./results/model_media/model_stats.csv", title_size=24, axis_label_size=20):
    """Plots model results as a 2x3 figure of the MSE and cindex of the 3 datasets"""
    df = prepare_df(stats_csv)

    models = {
        'DG': ('nomsa', 'binary', 'original', 'binary'),
        'esm': ('ESM', 'binary', 'original', 'binary'), # esm model
        'aflow': ('nomsa', 'aflow', 'original', 'binary'),
        # 'gvpP': ('gvp', 'binary', 'original', 'binary'),
        'gvpL': ('nomsa', 'binary', 'gvp', 'binary'),
        # 'aflow_ring3': ('nomsa', 'aflow_ring3', 'original', 'binary'),
        'gvpL_aflow': ('nomsa', 'aflow', 'gvp', 'binary'),
        # 'gvpl_esm':('ESM', 'binary', 'gvp', 'binary'),
        # 'gvpL_aflow_rng3': ('nomsa', 'aflow_ring3', 'gvp', 'binary'),
        #GVPL_ESMM_davis3D_nomsaF_aflowE_48B_0.00010636872718329864LR_0.23282479481785903D_2000E_gvpLF_binaryLE
        # 'gvpl_esm_aflow': ('ESM', 'aflow', 'gvp', 'binary'),
    }


    fig, axes = fig_combined(df, datasets=['davis', 'kiba', 'PDBbind'], fig_callable=custom_fig,
                models=models, metrics=['cindex', 'mse'],
                fig_scale=(10,5), add_stats=True, 
                title_postfix=" test set performance", box=True, 
                fold_labels=False)
    for i in range(3):
        axes[0][i].title.set_size(title_size)
        axes[1][i].xaxis.get_label().set_size(axis_label_size)
        
    # # remove xaxis 0 and 2 labels
    # for i in [0,2]:
    #     axes[1][i].set_xlabel('') # how do I delete this label
    
    for i in range(2):
        axes[i][0].yaxis.get_label().set_size(axis_label_size)
    plt.tight_layout(pad=2)


##########################################################
#%% FIG 3 - PLATINUM DATASET - RESULTS
##########################################################
#%% Table for counts for number of unique ligands and proteins
def get_Platinum_dataset_counts():
    from src.utils.loader import Loader
    from src import cfg

    import logging
    logging.getLogger().setLevel(logging.WARNING)
    db = Loader.load_dataset(cfg.DATA_OPT.platinum,
                        pro_feature=cfg.PRO_FEAT_OPT.nomsa, 
                        edge_opt=cfg.PRO_EDGE_OPT.binary,
                        max_seq_len=21000)

    print("Platinum Dataset details:")
    print("\tUnique protein sequence counts:", len(db.df.prot_seq.unique()))
    print("\t            Unique protein IDs:", len(db.df.prot_id.str.split("_").str[0].unique()))
    print("\t          Unique ligand counts:", len(db.df.lig_id.unique()))
    print("\t                 Total records:", len(db.df))
    return db

def plot_Platinum_mutations_dist():
    df = get_Platinum_dataset_counts().df
    df['pdb_id'] = df.prot_id.str.split("_").str[0]
    df['n_muts'] = df.prot_id.str.split("-").str.len() # for prot_ids with "_wt" they should be set to zero 
    df.loc[df.prot_id.str.contains("_wt", na=False), 'n_muts'] = 0

    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np

    df_view = df#[df['n_muts'] > 0] # to limit it to just mutated proteins add this
    bin_edges = np.arange(df_view['n_muts'].min() - 0.5, df_view['n_muts'].max() + 1.5, 1)
    plt.figure(figsize=(10,5))
    sns.histplot(df_view['n_muts'], bins=bin_edges)
    plt.title("Number of mutations per protein in Platinum")
    plt.xlabel("Number of mutations")
    plt.ylabel("Frequency")
    plt.tight_layout()

#%%
def plot_Platinum_delta_pkd_distribution_by_mutation_count():
    df = get_Platinum_dataset_counts().df
    df['pdb_id'] = df.prot_id.str.split("_").str[0]
    df['n_muts'] = df.prot_id.str.split("-").str.len()  # Number of mutations
    df.loc[df.prot_id.str.contains("_wt", na=False), 'n_muts'] = 0  # Set n_muts to 0 for wild type

    # Separate wild-type and mutated proteins
    df_wt = df[df['prot_id'].str.contains('_wt', na=False)].copy()
    df_mut = df[~df['prot_id'].str.contains('_wt', na=False)].copy()

    # Merge to compute delta pkd
    delta_df = pd.merge(
        df_mut,
        df_wt[['pdb_id', 'pkd']],
        on='pdb_id',
        suffixes=('_mut', '_wt')
    )

    # Calculate delta pkd
    delta_df['delta_pkd'] = delta_df['pkd_mut'] - delta_df['pkd_wt']

    # Group by number of mutations
    delta_1_mut = delta_df[delta_df['n_muts'] == 1]
    delta_2_mut = delta_df[delta_df['n_muts'] == 2]
    delta_3plus_mut = delta_df[delta_df['n_muts'] >= 3]

    # Plot overlayed distributions
    import seaborn as sns
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 6))
    from functools import partial
    shist = partial(sns.histplot, kde=True, bins=30, alpha=0.3, 
                    line_kws={'linewidth': 3, 'linestyle': 'solid'},
                    linewidth=2, edgecolor=None, stat="density")
    
    shist(delta_1_mut['delta_pkd'],     label="1 Mutation",  color="skyblue")
    shist(delta_2_mut['delta_pkd'],     label="2 Mutations", color="lightgreen")
    shist(delta_3plus_mut['delta_pkd'], label="3+ Mutations",color="orange")
    
    plt.title(r"Distribution of $\Delta pkd$ by Mutation Count")
    plt.xlabel(r"$\Delta pkd$")
    plt.ylabel("Density")
    plt.legend(title="Mutation Count")
    plt.tight_layout()
    plt.show()

#%%
def plot_Platinum_pkd_distribution():
    df = get_Platinum_dataset_counts().df
    df['pdb_id'] = df.prot_id.str.split("_").str[0]
    df['n_muts'] = df.prot_id.str.split("-").str.len()  # Number of mutations
    df.loc[df.prot_id.str.contains("_wt", na=False), 'n_muts'] = 0  # Set n_muts to 0 for wild type

    # Categorize data
    df['Category'] = '3+ Mutations'
    df.loc[df['n_muts'] == 1, 'Category'] = '1 Mutation'
    df.loc[df['n_muts'] == 2, 'Category'] = '2 Mutations'
    df.loc[df['n_muts'] == 0, 'Category'] = 'Wildtype'

    # Plot the distributions of pkd
    import seaborn as sns
    import matplotlib.pyplot as plt
    from functools import partial
    shist = partial(sns.histplot, x='pkd', kde=True, bins=30, alpha=0.3, 
                    line_kws={'linewidth': 3, 'linestyle': 'solid'},
                    linewidth=2, edgecolor=None, stat="density")

    plt.figure(figsize=(12, 6))
    
    # Plot each category
    shist(data=df[df['Category'] == 'Wildtype'], label="Wildtype", color='gray')
    shist(data=df[df['Category'] == '1 Mutation'],label="1 Mutation", color='skyblue')
    shist(data=df[df['Category'] == '2 Mutations'],label="2 Mutations", color='lightgreen')
    shist(data=df[df['Category'] == '3+ Mutations'],label="3+ Mutations", color='orchid')
    
    plt.title("Distribution of $pkd$ for Wildtype and Mutated Proteins")
    plt.xlabel("$pkd$")
    plt.ylabel("Density")
    plt.legend(title="Protein Category")
    plt.tight_layout()
    plt.show()

#%%
# FIG 3 - Platinum MODEL RESULTS
###############################
# - Run all 5 models through platinum and save predicted pkds as platinum_preds/<model_opt>_<fold>.csv
def Platinum_run_inference():
    """
    This script runs inference on platinum for the following models:
            ['davis_DG',    'davis_gvpl',   'davis_esm', 
            'kiba_DG',     'kiba_esm',     'kiba_gvpl',
            'PDBbind_DG',  'PDBbind_esm',  'PDBbind_gvpl', 
            'PDBbind_gvpl_aflow']
    
    It assumes that checkpoints for these models are already present in the CHECKPOINT_SAVE_DIR location
    as specified by the src/utils/config file.
    
    Also for any aflow model it also requires that aflow predictions for structures have already been generated
    otherwise building that dataset will be impossible.
    
    If the dataset is already built for that model then no building of the dataset will be done so long as they
    are in the right location as specified by DATA_ROOT in the src/utils/config file.
    """
    import logging
    import os

    import torch
    import pandas as pd
    from src.utils.loader import Loader
    from src import TUNED_MODEL_CONFIGS, cfg
    from collections import defaultdict
    from tqdm import tqdm
    logging.getLogger().setLevel(logging.INFO)

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    MODEL, model_kwargs =  Loader.load_tuned_model('davis_esm', fold=0, device=DEVICE)


    model_opts = [
                    'davis_DG',    'davis_esm',    'davis_gvpl',   'davis_aflow',   'davis_gvpl_aflow', 
                    'kiba_DG',     'kiba_esm',     'kiba_gvpl',    'kiba_aflow',    'kiba_gvpl_aflow',
                    'PDBbind_DG',  'PDBbind_esm',  'PDBbind_gvpl', 'PDBbind_aflow', 'PDBbind_gvpl_aflow'
                ]
    for model_opt in model_opts:
        loader = None
        for fold in range(5):
            print(f"{model_opt}-{fold}")
            out_csv = f"./results/platinum_predictions/{model_opt}_{fold}.csv"
            if os.path.exists(out_csv):
                print('\t Predictions already exists')
                continue
            
            MODEL_PARAMS = TUNED_MODEL_CONFIGS[model_opt]
            MODEL, model_kwargs =  Loader.load_tuned_model(model_opt, fold=fold, device=DEVICE)
            MODEL.eval()
            print("\t Model loaded")

            if loader is None: # caches loader if already created for this model_opt
                loader = Loader.load_DataLoaders(
                                data=cfg.DATA_OPT.platinum,
                                datasets=['full'],
                                pro_feature=MODEL_PARAMS['feature_opt'],
                                edge_opt=MODEL_PARAMS['edge_opt'],
                                ligand_feature=MODEL_PARAMS['lig_feat_opt'],
                                ligand_edge=MODEL_PARAMS['lig_edge_opt'],
                                )['full']
            print("\t Dataset loaded")


            PREDICTIONS = defaultdict(list)
            for batch in tqdm(loader, desc="\t running inference", ncols=100):    
                PREDICTIONS['code'].extend(batch['code'])
                PREDICTIONS['y'].extend(batch['y'].tolist())
                y_pred = MODEL(batch['protein'].to(DEVICE), batch['ligand'].to(DEVICE))
                PREDICTIONS['y_pred'].extend(y_pred[:,0].tolist())
                
                

            df = pd.DataFrame.from_dict(PREDICTIONS)
            df.set_index('code', inplace=True)
            df.sort_index(key = lambda x: x.str.split("_").str[0].astype(int), inplace=True)
            df.to_csv(out_csv)

    print("DONE!")

def platinum_fix_missing_pkd_vals():
    """
    Due to error in platinum raw csv parsing some values were dropped 
    (see https://github.com/jyaacoub/MutDTA/pull/148/commits/5611c8146bdc4b2ff67a60ea6f5d8f527d66d9db)
    
    This is a data patch to fix those dropped values if we encounter them.
    """
    import pandas as pd
    import os
    root_dir = "/home/jean/projects/MutDTA/results/platinum_predictions"

    df_raw = pd.read_csv('/home/jean/projects/data/PlatinumDataset/raw/platinum_flat_file.csv', index_col=0)
    # fixing pkd values for binding affinity
    df_raw['affin.k_mt'] = df_raw['affin.k_mt'].str.extract(r'[<>=]*(.*\d+)', expand=False).astype(float)
    # adjusting units for binding data from nM to pKd:
    df_raw['affin.k_mt'] = -np.log10(df_raw['affin.k_mt']*1e-9)
    df_raw['affin.k_wt'] = -np.log10(df_raw['affin.k_wt']*1e-9)

    for f in os.listdir(root_dir):
        fp = os.path.join(root_dir, f)
        df = pd.read_csv(fp, index_col=0)
        nan_found = False
        for code in df[df['y'].isna()].index:
            nan_found = True
            i, mt_wt = code.split('_')
            i = int(i)
            print(code, end=' - ')
            if mt_wt == 'mt':
                df.loc[code, 'y'] = df_raw.iloc[i]['affin.k_mt']
            else:
                df.loc[code, 'y'] = df_raw.iloc[i]['affin.k_wt']
            print(df.loc[code]['y'])

        if nan_found: df.to_csv(fp)

def get_all_folds_df(pred_csv=lambda model_opt, fold: f"./results/platinum_predictions/{model_opt}_{fold}.csv", 
                     model_opt='davis_DG'):
    """
    Gets all predictions for a model on platinum dataset returning dataframe like the following:
                y	y_pred_0	y_pred_1	y_pred_2	y_pred_3	y_pred_4	y_pred_avg
        code							
        0_wt	9.000000	5.150848	5.298963	5.935942	5.260733	5.207066	5.370710
        0_mt	8.494850	5.194075	5.409072	6.133554	5.289364	5.201524	5.445518
        1_mt	8.886057	5.168861	5.338132	5.817824	5.299617	5.189447	5.362776
    """
    all_folds = pd.read_csv(pred_csv(model_opt, 0), index_col='code')
    for fold in range(1,5):
        new_fold = pd.read_csv(pred_csv(model_opt, fold), index_col='code')[['y_pred']]
        all_folds = all_folds.join(new_fold, on='code', rsuffix=f'_{fold}')

    all_folds.rename(columns={'y_pred': 'y_pred_0'}, inplace=True)
    all_folds['y_pred_avg'] = all_folds[[f'y_pred_{i}' for i in range(5)]].mean(axis=1)
    return all_folds
    
#%%
def platinum_pkd_model_results(pred_csv=
                               lambda model_opt, fold: f"./results/platinum_predictions/{model_opt}_{fold}.csv",
                               model_opts =['davis_DG',    'davis_gvpl',   'davis_esm', 
                                            'kiba_DG',     'kiba_esm',     'kiba_gvpl',
                                            'PDBbind_DG',  'PDBbind_esm',  'PDBbind_gvpl', 
                                            'PDBbind_gvpl_aflow'],
                               normalized=True,
                               subset:list[str]=[],
                               DELTA=False): # subset of platinum indicies to apply metrics to (useful for stratified results like "in or out of pocket" mutations)
    """
    NOTE: CINDEX AND PEARSON WILL NOT BE IMPACTED BY NORMALIZATION
    
    If DELTA is set to True then this gets the "model's ability to predict the CHANGE in binding affinity"
    
    OTHERWISE it gets the "RAW predictive performance on platinum"
    
    Gets metrics for models across all 5 folds for each model
    Creates a dataframe replicating the "results/model_media/models_stats.csv" format:
                                                          run	  cindex	  pearson	 spearman	     mse	...	improved	batch_size	     lr	dropout	overlap
        0	DGM_davis0D_nomsaF_binaryE_128B_0.00012LR_0.24...	0.359369	-0.280921	-0.430664	7.112641	...	   False	       128	0.00012	   0.24	  False
        1	DGM_davis1D_nomsaF_binaryE_128B_0.00012LR_0.24...	0.426336	-0.229465	-0.225258	6.071823	...	   False	       128	0.00012	   0.24	  False
        2	DGM_davis2D_nomsaF_binaryE_128B_0.00012LR_0.24...	0.521040	 0.098562	 0.060612	5.227944	...	   False	       128	0.00012	   0.24	  False
    """
    import pandas as pd
    from src.utils.loader import Loader
    from src import TUNED_MODEL_CONFIGS
    from src.analysis.metrics import get_metrics
    from src.analysis.figures import prepare_df

    metrics = {'run': [],'cindex': [],'pearson': [],'spearman': [],'mse': [],'mae': [],'rmse': []}
    for model_opt in model_opts:
        all_folds = get_all_folds_df(pred_csv, model_opt)
        # normalize
        if normalized:
            #z-score norm
            all_folds = (all_folds - np.mean(all_folds, axis=0)) / np.std(all_folds, axis=0)
        
        if DELTA:
            # Calculate DELTA_pkd >>>>
            all_folds['pro'] = all_folds.index.str.extract(r'(\d+)_[wm]t', expand=False)
            all_folds_wt = all_folds[all_folds.index.str.contains('wt')]
            
            all_folds_reset = all_folds.reset_index() # to maintain index we reset before merge to make it a column
            all_folds = all_folds_reset.merge(all_folds_wt, how="left", on="pro", suffixes=("_mt", "_wt"))
            all_folds.set_index('code', inplace=True) # added back index
            
            # doing subtraction to get delta values:
            mt_cols = [col for col in all_folds.columns if '_mt' in col]
            wt_cols = [col.replace('_mt', '_wt') for col in mt_cols]
            
            all_folds = all_folds[wt_cols].sub(all_folds[mt_cols].values, axis=0)
            
            # rename back to original table since the rest of the code is the same as RAW_pkd method
            # dropping the _mt suffix
            all_folds.rename(columns={col: col[:-3] for col in all_folds.columns}, inplace=True)
            
            assert len(all_folds) == 1962, f"Missing rows in pred csv, {len(all_folds)}/1962 for {model_opt}"
            
            # dropping wt rows since those will be all zeros
            all_folds = all_folds[all_folds.index.str.contains('_mt')]
            # <<<<
        else:
            assert len(all_folds) == 1962, f"Missing rows in pred csv, {len(all_folds)}/1962 for {model_opt}"
            
        
        # taking subset if any
        if subset:
            try:
                all_folds = all_folds.loc[subset]
            except KeyError as e:
                err_str = "DataFrame failed to retrieve elements from subset."
                if DELTA:
                    err_str += " NOTE: wt with DELTA is not allowed since that would just result in zeros"
                raise KeyError(err_str) from e
                
        
        for fold in range(5):
            def reformat_kwargs(model_kwargs):
                return {
                    'model': model_kwargs['model'],
                    'data': model_kwargs['dataset'],
                    'pro_feature': model_kwargs['feature_opt'],
                    'edge': model_kwargs['edge_opt'],
                    'batch_size': model_kwargs['batch_size'],
                    'lr': model_kwargs['lr'],
                    'dropout': model_kwargs['architecture_kwargs']['dropout'],
                    'n_epochs': model_kwargs.get('n_epochs', 2000),  # Assuming a default value for n_epochs
                    'pro_overlap': model_kwargs.get('pro_overlap', False),  # Assuming a default or None
                    'fold': model_kwargs.get('fold', fold),  # Assuming a default or None
                    'ligand_feature': model_kwargs['lig_feat_opt'],
                    'ligand_edge': model_kwargs['lig_edge_opt']
                }

            model_kwargs = reformat_kwargs(TUNED_MODEL_CONFIGS[model_opt])
            run = Loader.get_model_key(**model_kwargs)
            metrics['run'].append(run)
            
            # create metrics df for plotting results
            cindex, p_corr, s_corr, mse, mae, rmse = get_metrics(all_folds['y'], all_folds[f'y_pred_{fold}'])
            metrics['cindex'].append(cindex)
            metrics['pearson'].append(p_corr[0])
            metrics['spearman'].append(s_corr[0])
            metrics['mse'].append(mse)
            metrics['mae'].append(mae)
            metrics['rmse'].append(rmse)
        
    df_metrics = pd.DataFrame.from_dict(metrics)
    prepare_df(df=df_metrics)
    return df_metrics

def platinum_RAW_pkd_model_results(*args, **kwargs) -> pd.DataFrame:
    """
    Returns pandas dataframe for df input to `custom_fig` and similar figures methods
    """
    return platinum_pkd_model_results(*args, **kwargs, DELTA=False)

def platinum_DELTA_pkd_model_results(*args, **kwargs) -> pd.DataFrame:
    """
    Returns pandas dataframe for df input to `custom_fig` and similar figures methods
    """
    return platinum_pkd_model_results(*args, **kwargs, DELTA=True)

#%%
def platinum_mt_in_pocket_indicies(raw_csv='/home/jean/projects/data/PlatinumDataset/raw/platinum_flat_file.csv'):
    """
    df_raw['mut.in_binding_site'].value_counts()
        YES    725
        NO     256
        Name: mut.in_binding_site, dtype: int64
    
    returns 3 lists corresponding to the following
        0. wildtype rows
        1. in pocket
        2. outside of pocket
    """
    wt = []
    in_pocket = []
    out_pocket = []
    df_raw = pd.read_csv(raw_csv, index_col=0)
    for i, row in df_raw.iterrows():
        wt.append(f'{i}_wt')
        
        if row['mut.in_binding_site'] == 'YES':
            in_pocket.append(f'{i}_mt')
        else:
            out_pocket.append(f'{i}_mt')
            
    return wt, in_pocket, out_pocket

def resampling(
    subset_groups: Dict[str, List],
    callable_pkd_model_results: Callable[[List], pd.DataFrame],
    num_samples: int = 10,
) -> Dict[str, pd.DataFrame]:
    """
    Calculate averaged results over multiple random samples for multiple groups.
    
    Args:
        subset_groups: Dictionary where keys are group names and values are lists of indices.
        callable_pkd_model_results: Callable that computes metrics for a given subset.
        num_samples: Number of random samples to average over.

    Returns:
        Dictionary with averaged metrics for each group.
    """

    random.seed(42)

    # Determine the maximum resampling size (smallest group size)
    max_size = min(len(subset) for subset in subset_groups.values())
    all_metrics = {group: [] for group in subset_groups}

    # Generate multiple samples and compute metrics
    for _ in tqdm(range(num_samples), desc="Resampling"):
        for group_name, subset in subset_groups.items():
            # if max_size == len(subset) and len(all_metrics[group_name]) > 0:
            #     # only need to do this once since there is only one possible sampling
            #     continue
            # Resample
            sampled_subset = random.sample(subset, max_size)

            # Compute metrics for the current sample
            metrics = callable_pkd_model_results(subset=sampled_subset)
            all_metrics[group_name].append(metrics)

    # Average the metrics across all samples for each group
    averaged_metrics = {}
    columns_to_average = ['cindex', 'pearson', 'spearman', 'mse', 'mae', 'rmse']

    for group_name, metrics_list in all_metrics.items():
        # Concatenate metrics for the current group
        df_concat = pd.concat(metrics_list)

        # Average only the selected columns
        numeric_avg = df_concat[columns_to_average].groupby(level=0).mean(numeric_only=True)

        # For all other columns, just pick the first value
        non_averaged_columns = df_concat.columns.difference(columns_to_average)
        non_numeric_first = df_concat[non_averaged_columns].groupby(level=0).first()

        # Combine the averaged and non-averaged columns back together
        averaged_metrics[group_name] = pd.concat([non_numeric_first, numeric_avg], axis=1)

    return averaged_metrics

#%%
# mt_in is a larger subset than mt_out so we need to do some resampling to ensure that the 
# size of the dataset doesnt impact metrics
wt, mt_in, mt_out = platinum_mt_in_pocket_indicies()

subset_groups = {
    "wt": wt,
    "mt_in": mt_in,
    "mt_out": mt_out
}

# %%
averaged_results = resampling(
    subset_groups={k:v for k,v in subset_groups.items() if k != 'wt'},
    callable_pkd_model_results=platinum_DELTA_pkd_model_results,
    num_samples=10
)

# %%
averaged_results = resampling(
    subset_groups=subset_groups,
    callable_pkd_model_results=platinum_RAW_pkd_model_results,
    num_samples=10
)
# %%
