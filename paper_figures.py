#%%
import pandas as pd
import matplotlib.pyplot as plt

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


    model_opts = ['davis_DG',    'davis_gvpl',   'davis_esm', 
                'kiba_DG',     'kiba_esm',     'kiba_gvpl',
                'PDBbind_DG',  'PDBbind_esm',  'PDBbind_gvpl', 
                'PDBbind_gvpl_aflow']
    for model_opt in model_opts:
        loader = None
        for fold in range(5):
            print(f"{model_opt}-{fold}")
            out_csv = f"./results/platinum_predictions/{model_opt}_{fold}.csv"
            if os.path.exists(out_csv):
                print('\t Predictions already exists')
                continue
            
            MODEL_PARAMS = TUNED_MODEL_CONFIGS[model_opt]
            try:
                MODEL, model_kwargs =  Loader.load_tuned_model(model_opt, fold=fold, device=DEVICE)
            except AssertionError as e:
                print(e)
                continue
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
    import numpy as np
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
        for code in df[df['y'].isna()].index:
            i, mt_wt = code.split('_')
            i = int(i)
            print(code, end=' - ')
            if mt_wt == 'mt':
                df.loc[code, 'y'] = df_raw.iloc[i]['affin.k_mt']
            else:
                df.loc[code, 'y'] = df_raw.iloc[i]['affin.k_wt']
            print(df.loc[code]['y'])
        df.to_csv(fp)
    

def get_all_folds_df(pred_csv=lambda model_opt, fold: f"./results/platinum_predictions/{model_opt}_{fold}.csv", model_opt='davis_DG'):
    all_folds = pd.read_csv(pred_csv(model_opt, 0), index_col='code')
    for fold in range(1,5):
        new_fold = pd.read_csv(pred_csv(model_opt, fold), index_col='code')[['y_pred']]
        all_folds = all_folds.join(new_fold, on='code', rsuffix=f'_{fold}')

    all_folds.rename(columns={'y_pred': 'y_pred_0'}, inplace=True)
    all_folds['y_pred_avg'] = all_folds[[f'y_pred_{i}' for i in range(5)]].mean(axis=1)
    return all_folds
    