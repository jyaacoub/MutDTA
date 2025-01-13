from collections import Counter, OrderedDict
import logging
import os, pickle, json
import functools

import pandas as pd
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from statannotations.Annotator import Annotator
from scipy.stats import ttest_ind
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from src.utils import config as cfg
from src.utils.loader import Loader
from src.analysis.metrics import get_metrics
from src.analysis.utils import generate_markdown, get_mut_count


def fig0_dataPro_overlap(data:str='davis', data_root:str=cfg.DATA_ROOT, verbose=False):
    data_path = f'{data_root}/{data}'
    
    Y = pickle.load(open(f'{data_path}/Y', "rb"), encoding='latin1')
    row_i, col_i = np.where(np.isnan(Y)==False)
    test_fold = json.load(open(f"{data_path}/folds/test_fold_setting1.txt"))
    train_fold = json.load(open(f"{data_path}/folds/train_fold_setting1.txt"))
    
    # loading up train and test protein indices
    train_flat = [i for fold in train_fold for i in fold]
    test_protein_indices = col_i[test_fold]
    train_protein_indices = col_i[train_flat]

    # Overlap in train and test...
    overlap = set(train_protein_indices).intersection(set(test_protein_indices))
    if verbose:
        print(f'number of unique proteins in train: {len(set(train_protein_indices))}')
        print(f'number of unique proteins in test:  {len(set(test_protein_indices))}')
        print(f'total number of unique proteins:    {max(col_i)+1}')
        print(f'Intersection of train and test protein indices: {len(overlap)}')

    # counts of overlaping proteins
    test_counts = Counter(test_protein_indices)
    train_counts = Counter(train_protein_indices)

    overlap_test_counts = {k: test_counts[k] for k in overlap}
    overlap_train_counts = {k: train_counts[k] for k in overlap}

    # normalized for set size
    norm_overlap_test_counts = {k: v/len(test_protein_indices) for k,v in overlap_test_counts.items()}
    norm_overlap_train_counts = {k: v/len(train_protein_indices) for k,v in overlap_train_counts.items()}

    # plot overlap counts
    plt.figure(figsize=(15,10))
    plt.subplot(2,1,1)
    plt.bar(overlap_train_counts.keys(), overlap_train_counts.values(), label='train', width=1.0)
    plt.bar(overlap_test_counts.keys(), overlap_test_counts.values(), label='test', width=1.0)
    plt.xlabel('protein index')
    plt.ylabel('count')
    plt.title(f'Counts of proteins in train and test ({data})')
    plt.legend()

    plt.subplot(2,1,2)
    plt.bar(norm_overlap_train_counts.keys(), norm_overlap_train_counts.values(), label='train', width=1.0)
    plt.bar(norm_overlap_test_counts.keys(), norm_overlap_test_counts.values(), label='test', width=1.0)
    plt.xlabel('protein index')
    plt.ylabel('Normalized Counts')
    plt.title(f'Normalized counts by dataset size of proteins in train and test ({data})')
    plt.legend()
    plt.tight_layout()

# Figure 1 - Protein overlap cindex difference (nomsa)
def fig1_pro_overlap(df, sel_col='cindex', verbose=False, show=False, context='paper'):
    filtered_df = df[(df['feat'] == 'nomsa') 
                    & (df['batch_size'] == '64') 
                    & (df['edge'] == 'binary')
                    & (~df['ddp'])              
                    & (~df['improved'])]
    
    #
    plot_df = filtered_df[['data', 'overlap', sel_col]]
    if context == 'poster':
        scale = 1
        plt.figure(figsize=(14, 7))
    else:
        scale = 0.8
        plt.figure(figsize=(10, 5))
    
    sns.set(style="darkgrid")
    sns.set_context(context)
    hue_order = [True, False]
    ax = sns.barplot(data=plot_df, x='data', y=sel_col, hue='overlap', palette='deep', estimator=np.mean,
                     order=['PDBbind', 'davis', 'kiba'], hue_order=hue_order, errwidth=0)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, title='Overlap', loc='upper right', prop={'size': 14*scale})
    
    # Set the y-axis label
    ax.set_ylabel(sel_col)
    if sel_col == 'cindex':
        ax.set_ylim([0.5, 1]) # 0.5 is the worst cindex value
    ax.set_xlabel('Dataset')
    # Set the title and legend
    ax.set_title(f'Protein Overlap {sel_col} Difference (DGraphDTA)',
                 fontsize=16*scale)

    # Show the plot
    if show: plt.show()
    
    return plot_df

# Figure 2 - node feature cindex difference
# Features -> nomsa, msa, shannon, and esm
def fig2_pro_feat(df, verbose=False, sel_col='cindex', exclude=[], show=False, add_labels=True,
                  context='poster'):
    # Extract relevant data
    filtered_df = df[(df['edge'] == 'binary') & (~df['overlap']) 
                     & (df['fold'] != '') & (df['lig_feat'] == 'original')]
    
    # get only data, feat, and sel_col columns
    plot_df = filtered_df[['data', 'feat', sel_col]]
        
    hue_order = ['nomsa', 'msa', 'shannon', 'ESM']
    for f in exclude:
        plot_df = plot_df[plot_df['feat'] != f]
        if f in hue_order:
            hue_order.remove(f)   
    
    # Create a bar plot using Seaborn
    if context == 'poster':
        scale = 1
        plt.figure(figsize=(14, 7))
    else:
        scale = 0.8
        plt.figure(figsize=(10, 5))

    sns.set(style="darkgrid")
    sns.set_context(context)
    ax = sns.barplot(data=plot_df, x='data', y=sel_col, hue='feat', palette='deep', estimator=np.mean,
                     order=['PDBbind', 'davis', 'kiba'], hue_order=hue_order, errcolor='gray', errwidth=2)
    sns.stripplot(data=plot_df, x='data', y=sel_col, hue='feat', palette='deep',
                  order=['PDBbind', 'davis', 'kiba'], hue_order=hue_order,
                  size=6*scale, jitter=True, dodge=True, alpha=0.7, ax=ax)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[len(hue_order):], labels[len(hue_order):], 
              title='', loc='upper right', prop={'size': 14*scale})
    
    if add_labels:
        for i in ax.containers: 
            ax.bar_label(i, fmt='%.3f', fontsize=13*scale, 
                         label_type='center')
            
    # Set the title
    ax.set_title(f'Node feature performance ({"concordance index" if sel_col == "cindex" else sel_col})',
                 fontsize=16*scale)
    
    # Set the y-axis label and limit
    ax.set_ylabel(sel_col)
    if sel_col == 'cindex':
        ax.set_ylim([0.5, 1])  # 0.5 is the worst cindex value
    if sel_col == 'pearson':
        ax.set_ylim([0, 1])
    
    # Add statistical annotations
    pairs=[("PDBbind", "kiba"), ("PDBbind", "davis"), ("davis", "kiba")]
    annotator = Annotator(ax, pairs, data=plot_df, 
                          x='data', y=sel_col, order=['PDBbind', 'davis', 'kiba'], #NOTE: this needs to be fixed
                          verbose=verbose)
    annotator.configure(test='Mann-Whitney', text_format='star', loc='inside', hide_non_significant=True,
                        verbose=verbose)
    annotator.apply_and_annotate()
    
    # Show the plot
    if show:
        plt.show()
        
    # reset stylesheet back to defaults
    mpl.rcParams.update(mpl.rcParamsDefault)
    
    return plot_df

# Figure 3 - Edge type cindex difference
# Edges -> binary, simple, anm, af2
def fig3_edge_feat(df, verbose=False, sel_col='cindex', exclude=['af2-anm'], show=False, add_labels=True,
                   context='poster'):
    # comparing nomsa, msa, shannon, and esm
    # group by data type
    
    # this will capture multiple models per dataset (different LR, batch size, etc)
    #   Taking the max cindex value for each dataset will give us the best model for each dataset
    filtered_df = df[(df['feat'] == 'nomsa') & (~df['overlap']) 
                     & (df['fold'] != '') & (df['lig_feat'] == 'original')]
    plot_df = filtered_df[['data', 'edge', sel_col]]
    
    hue_order = ['binary', 'simple', 'anm', 'af2', 'af2-anm']
    for f in exclude:
        plot_df = plot_df[plot_df['edge'] != f]
        if f in hue_order:
            hue_order.remove(f)    
    
    # Create a bar plot using Seaborn
    scale = 0.8 if context == 'paper' else 1
    if context == 'poster':
        plt.figure(figsize=(14, 7))
    else:
        plt.figure(figsize=(10, 5))
        
    sns.set(style="darkgrid")
    sns.set_context(context)
    ax = sns.barplot(data=plot_df, x='data', y=sel_col, hue='edge', palette='deep', estimator=np.mean,
                     order=['PDBbind', 'davis', 'kiba'], hue_order=hue_order, errcolor='gray', errwidth=2)
    sns.stripplot(data=plot_df, x='data', y=sel_col, hue='edge', palette='deep',
                  order=['PDBbind', 'davis', 'kiba'], hue_order=hue_order,
                  size=6*scale, jitter=True, dodge=True, alpha=0.7, ax=ax)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[len(hue_order):], labels[len(hue_order):], prop={'size': 14*scale},
              title='', loc='upper right')
    
    if add_labels:
        for i in ax.containers: 
            ax.bar_label(i, fmt='%.3f', fontsize=13*scale, label_type='center')
            
    # Set the title
    ax.set_title(f'Edge type performance ({"concordance index" if sel_col == "cindex" else sel_col})',
                 fontsize=16*scale)

    # Set the y-axis label and limit
    ax.set_ylabel(sel_col)
    if sel_col == 'cindex':
        ax.set_ylim([0.5, 1])  # 0.5 is the worst cindex value
    if sel_col == 'pearson':
        ax.set_ylim([0, 1])
        
    # Add statistical annotations
    pairs=[("PDBbind", "kiba"), ("PDBbind", "davis"), ("davis", "kiba")]
    annotator = Annotator(ax, pairs, data=plot_df, 
                          x='data', y=sel_col, order=['PDBbind', 'davis', 'kiba'],
                          verbose=verbose)
    annotator.configure(test='Mann-Whitney', text_format='star', loc='inside', hide_non_significant=True,
                        verbose=verbose)
    annotator.apply_and_annotate()
    # Show the plot
    if show:
        plt.show()
        
    # reset stylesheet back to defaults
    mpl.rcParams.update(mpl.rcParamsDefault)
    
    return plot_df

# Figure 4: violin plot with error bars for Cross-validation results to show significance among pro feats
def fig4_pro_feat_violin(df, sel_dataset='davis', verbose=False, sel_col='cindex', exclude=[], 
                         show=False, add_stats=True, ax=None):
    # Filter data based on conditions
    filtered_df = df[(df['edge'] == 'binary') & (~df['overlap']) & (df['lig_feat'] == 'original')]
    filtered_df = filtered_df[(filtered_df['data'] == sel_dataset) & (filtered_df['fold'] != '')]

    # Dynamically collect values for each feature type not in exclude
    features = filtered_df['feat'].unique()
    features = [f for f in features if f not in exclude]
    plot_data = [filtered_df[filtered_df['feat'] == feature][sel_col] for feature in features]

    # Dynamically plotting
    ax = sns.violinplot(data=plot_data, ax=ax)
    ax.set_xticklabels(features)
    ax.set_ylabel(sel_col)
    ax.set_xlabel('Protein Node Features')
    ax.set_title(f'Feature {sel_col} for {sel_dataset}')

    # Verbose printing
    if verbose:
        for feature, data in zip(features, plot_data):
            print(f'{feature}: {len(data)}')

    # Annotation for stats
    if add_stats:
        pairs = [(i, j) for i in range(len(features)) for j in range(i+1, len(features))]
        annotator = Annotator(ax, pairs, data=plot_data, verbose=verbose)
        annotator.configure(test='Mann-Whitney', text_format='star', loc='inside', 
                            hide_non_significant=not verbose)
        annotator.apply_and_annotate()

    if show:
        plt.show()
        
    return plot_data

# Figure 5: violin plot with error bars for Cross-validation results to show significance among edge feats
def fig5_edge_feat_violin(df, sel_dataset='davis', verbose=False, sel_col='cindex', exclude=[],
                            show=False, add_labels=True, add_stats=True, ax=None):
    filtered_df = df[(df['feat'] == 'nomsa') & (~df['overlap']) & (df['lig_feat'] == 'original')]
    filtered_df = filtered_df[(filtered_df['data'] == sel_dataset) & (filtered_df['fold'] != '')]

    filtered_df.sort_values(by=['edge'], inplace=True)

    # Dynamically collect values for each edge type not in exclude
    edge_types = filtered_df['edge'].unique()
    edge_types = [e for e in edge_types if e not in exclude]
    plot_data = [filtered_df[filtered_df['edge'] == edge][sel_col] for edge in edge_types]

    # plot violin plot with annotations
    ax = sns.violinplot(data=plot_data, ax=ax)
    ax.set_xticklabels(edge_types)
    ax.set_ylabel(sel_col)
    ax.set_xlabel('Edge type')
    ax.set_title(f'Edge type {sel_col} for {sel_dataset}')

    if add_stats:
        pairs = [(i, j) for i in range(len(edge_types)) for j in range(i+1, len(edge_types))]
        annotator = Annotator(ax, pairs, data=plot_data, verbose=verbose)
        annotator.configure(test='Mann-Whitney', text_format='star', loc='inside', 
                            hide_non_significant=not verbose)
        annotator.apply_and_annotate()
        
    if show:
        plt.show()
        
    return plot_data

def fig6_protein_appearance(datasets=['kiba', 'PDBbind'], show=False):
    # Create a subplot with 3 rows and 1 column
    fig, axs = plt.subplots(len(datasets), 1, figsize=(10, 5*len(datasets)))

    for i, data in enumerate(datasets):
        # Load the dataset
        dataset = Loader.load_dataset(data=data, pro_feature='nomsa', edge_opt='binary', subset='full')

        # Get counts for each protein in the dataset
        cntr = dataset.get_protein_counts()
        
        # Plot as histogram
        axs[i].hist(cntr.values(), bins=100)
        
        axs[i].set_yscale('log')
                
        axs[i].set_title(f'Frequency of Protein Appearance in {data} Dataset')
        axs[i].set_xlabel('Binned protein appearance count')
        axs[i].set_ylabel('Frequency (log scale)')
    # Adjust layout to prevent clipping of titles
    plt.tight_layout()
    
    if show: plt.show()
    
    
def fig_combined(df, datasets=['PDBbind','davis', 'kiba'], metrics=['cindex', 'mse'], 
                  fig_callable=fig4_pro_feat_violin, fig_scale=(5,4),
                  show=False, title_postfix='', **kwargs):    
    # Create subplots with datasets as columns and metrics as rows
    fig, axes = plt.subplots(len(metrics), len(datasets), 
                             figsize=(fig_scale[0]*len(datasets), 
                                      fig_scale[1]*len(metrics)))
    for i, dataset in enumerate(datasets):
        for j, metric in enumerate(metrics):
            # Set current subplot                
            if len(datasets) == 1 and len(metrics) == 1:
                ax = axes
            elif len(datasets) == 1:
                ax = axes[j]
            elif len(metrics) == 1:
                ax = axes[i]
            else:
                ax = axes[j, i]

            fig_callable(df, sel_col=metric, sel_dataset=dataset, show=False, 
                         ax=ax, **kwargs)
                        
            # Add titles only to the top row and left column
            if j == 0:
                ax.set_title(f'{dataset}{title_postfix}')
                ax.set_xlabel('')
                ax.set_xticklabels([])
            elif j < len(metrics)-1: # middle row
                ax.set_xlabel('')
                ax.set_xticklabels([])
                ax.set_title('')
            else: # bottom row
                ax.set_title('')
            
            if i == 0:
                ax.set_ylabel(metric)
            else:
                ax.set_ylabel('')
                
    plt.tight_layout() # Adjust layout to prevent clipping of titles
    if show: plt.show()
    return fig, axes

def custom_fig(df, models:OrderedDict=None, sel_dataset='PDBbind', sel_col='cindex', 
                   verbose=False, show=False, add_stats=True, ax=None, box=False, 
                   fold_points=True, fold_labels=False, alpha=0.7):
    
    """
    Example usage with `fig_combined`.
        ```
        from src.analysis.figures import custom_fig, prepare_df, fig_combined

        df = prepare_df()

        models = {
            'DG': ('nomsa', 'binary', 'original', 'binary'),
            # 'DG-simple': ('nomsa', 'simple', 'original', 'binary'),
            'DG-anm': ('nomsa', 'anm', 'original', 'binary'),
            'DG-af2': ('nomsa', 'af2', 'original', 'binary'),
            'DG-ESM': ('ESM', 'binary', 'original', 'binary'),
            # 'DG-saprot': ('foldseek', 'binary', 'original', 'binary'),
            'gvpP': ('gvp', 'binary', 'original', 'binary'),
            'gvpL-aflow': ('nomsa', 'aflow', 'gvp', 'binary'),
        }

        fig_combined(df, datasets=['PDBbind'], metrics=['cindex', 'mse'], fig_scale=(10,5),
                    fig_callable=custom_fig, models=models, title_postfix=' test set performance',
                    add_stats=True)
        ```
    """
    if models is None: # example custom plot:
        # models to plot:
        # - Original model with (nomsa, binary) and (original,  binary) features for protein and ligand respectively
        # - Aflow models with   (nomsa, aflow*) and (original,  binary) # x2 models here (aflow and aflow_ring3)
        # - GVP protein model   (gvp,   binary) and (original,  binary)
        # - GVP ligand model    (nomsa, binary) and (gvp,       binary)
        models = {
            'DG': ('nomsa', 'binary', 'original', 'binary'),
            'aflow': ('nomsa', 'aflow', 'original', 'binary'),
            'aflow_ring3': ('nomsa', 'aflow_ring3', 'original', 'binary'),
            'gvpP': ('gvp', 'binary', 'original', 'binary'),
            'gvpL': ('nomsa', 'binary', 'gvp', 'binary'),
        }
    
    # Filter out df based on args
    filtered_df = df[(df['data'] == sel_dataset) & (df['fold'] != '') & (~df['overlap'])]
    
    def matched(df, tuple):
        return (df['feat']      == tuple[0]) & (df['edge']     == tuple[1]) & \
                (df['lig_feat'] == tuple[2]) & (df['lig_edge'] == tuple[3])

    filter_conditions = [matched(filtered_df, v) for v in models.values()]
    filtered_df = filtered_df[sum(filter_conditions) > 0]

    # Group each model results
    all_data = OrderedDict()
    for model, feat in models.items():
        all_data[model] = filtered_df[matched(filtered_df, feat)][[sel_col, 'fold']]
        if len(all_data[model]) != 5:
            logging.warning(f'Expected 5 results for {model} on {sel_dataset}, got {len(all_data[model])}')

    # plot violin plot with annotations
    plot_data = OrderedDict({k: v[sel_col] for k, v in all_data.items()})
    fold_data = OrderedDict({k: v['fold'] for k, v in all_data.items()})
    folds = list(fold_data.values())
    vals = list(plot_data.values())
    if box:
        ax = sns.boxplot(data=vals, ax=ax, boxprops=dict(alpha=alpha))
    else:
        ax = sns.violinplot(data=vals, ax=ax)
        for violin in ax.collections:
            violin.set_alpha(alpha)

    if fold_points or fold_labels:
        sns.stripplot(data=vals, dodge=True, ax=ax, alpha=.8, linewidth=1)
        
        if fold_labels:
            adjs = -0.5 if box else -0.2
            for i in range(len(models)):
                for f, v in zip(folds[i], vals[i]):
                    ax.text(i+adjs, v, f, 
                            horizontalalignment='left', size='medium', color='red')
    
    ax.set_xticklabels(list(plot_data.keys()))
    ax.set_ylabel(sel_col)
    ax.set_xlabel('Model Type')
    ax.set_title(f'{sel_col} for {sel_dataset}')

    if add_stats:
        pairs = [(i, j) for i in range(len(models)) for j in range(i+1, len(models))]
        annotator = Annotator(ax, pairs, data=vals, verbose=verbose)
        annotator.configure(test='Mann-Whitney', text_format='star', loc='inside', 
                            hide_non_significant=not verbose)
        annotator.apply_and_annotate()
        
    if show:
        plt.show()
    
    return plot_data

def prepare_df(csv_p:str=cfg.MODEL_STATS_CSV, old_csv_p:str=None, df=None) -> pd.DataFrame:
    """
    Prepares a dataframe from the model stats csv file.

    Parameters
    ----------
    `csv_p` : str, optional
        Path to the model stats csv file, by default cfg.MODEL_STATS_CSV
    `old_csv_p` : str, optional
        Optional path to alternate csv file to merge stats with (e.g.: 
        "results/model_media/old_model_stats.csv"), by default None

    Returns
    -------
    pd.DataFrame
        - output dataframe with additional columns for easier filtering:
            - `data`: dataset type (davis, kiba, PDBbind)
            - `fold`: fold number if available
            - `feat`: node feature type (nomsa, msa, shannon, ESM)
            - `edge`: edge type (binary, simple, anm, af2, af2-anm)
            - `ddp`: whether or not the model was trained with DDP
            - `improved`: whether or not the model was trained with improved training
            - `batch_size`: batch size used for training
            - `overlap`: whether or not the model was trained with protein overlap 
    """
    if df is None:
        df = pd.read_csv(csv_p)
        
        if old_csv_p is not None and os.path.isfile(old_csv_p):
            df = pd.concat([df, pd.read_csv(old_csv_p)]) # concat with old model results since we get the max value anyways...

    # create data, feat, and overlap columns for easier filtering.
    df['data'] = df['run'].str.extract(r'_(davis|kiba|PDBbind)', expand=False)
    df['fold'] = df['run'].str.extract(r'_(davis|kiba|PDBbind)(\d*)', expand=True)[1] # fold number if available
    df['feat'] = df['run'].str.extract(r'_(nomsa|msa|shannon|foldseek|gvp)F_', expand=False)
    df['edge'] = df['run'].str.extract(r'_(binary|simple|anm|af2|af2_anm|ring3|aflow|aflow_ring3)E_', expand=False)
    
    # # ligand features and edges (defaults are original and binary):
    df['lig_feat'] = df['run'].str.extract(r'_(original|gvp)LF', expand=False).fillna('original')
    df['lig_edge'] = df['run'].str.extract(r'_(binary)LE', expand=False).fillna('binary')
    
    df['ddp'] = df['run'].str.contains('DDP-')
    df['improved'] = df['run'].str.contains('IM_') # postfix of model name will include I if "improved"
    df['batch_size'] = df['run'].str.extract(r'_(\d+)B_', expand=False).astype(int)
    
    df['lr'] = df['run'].str.extract(r'_(\d+\.?\d*[e-]*\d*)LR_', expand=False).astype(float)
    df['dropout'] = df['run'].str.extract(r'_(\d+\.?\d*)D_', expand=False).astype(float)
    
    # ESM models
    df.loc[df['run'].str.contains('ESM') & df['run'].str.contains('nomsaF'), 'feat'] = 'ESM'
    df.loc[df['run'].str.contains('EDM') & df['run'].str.contains('nomsaF'), 'feat'] = 'ESM'
    df.loc[df['run'].str.contains('EDAM'), 'feat'] += '-ESM'
    df.loc[df['run'].str.contains('EDIM') & df['run'].str.contains('nomsaF'), 'feat'] = 'ESM'
    df.loc[df['run'].str.contains('EDAIM'), 'feat'] += '-ESM'

    # ChemGPT models
    df.loc[df['run'].str.contains('CDM'), 'lig_feat'] = 'ChemGPT'
    #TODO: account for ChemGPT with additional features

    df['overlap'] = df['run'].str.contains('overlap')
    
    return df

#######################################################################################################
##################################### MUTATION ANALYSIS RELATED FIGS: #################################
#######################################################################################################
def predictive_performance(
    MODEL = lambda i: f"results/model_media/test_set_pred/GVPLM_PDBbind{i}D_nomsaF_aflowE_128B_0.00022659LR_0.02414D_2000E_gvpLF_binaryLE_PLATINUM.csv",
    TRAIN_DATA_P = lambda set: f'{cfg.DATA_ROOT}/PDBbindDataset/nomsa_aflow_gvp_binary/{set}0/cleaned_XY.csv',
    NORMALIZE = True,
    n_models=5,
    compare_overlap=False,
    verbose=True,
    plot=False,
    ):
    df_t = pd.Index.append(pd.read_csv(TRAIN_DATA_P('train'), index_col=0).index, 
                        pd.read_csv(TRAIN_DATA_P('val'), index_col=0).index)
    df_t = df_t.str.upper()

    results_with_overlap = []
    results_without_overlap = []

    for i in range(n_models):
        df = pd.read_csv(MODEL(i), index_col=0).dropna()
        df['pdb'] = df['prot_id'].str.split('_').str[0]
        if NORMALIZE:
            mean_df = df[['actual','pred']].mean(axis=0, numeric_only=True)
            std_df = df[['actual','pred']].std(axis=0, numeric_only=True)
            
            df[['actual','pred']] = (df[['actual','pred']] - mean_df) / std_df # z-normalization

        # with overlap
        cindex, p_corr, s_corr, mse, mae, rmse = get_metrics(df['actual'], df['pred'])
        results_with_overlap.append([cindex, p_corr[0], s_corr[0], mse, mae, rmse])

        # without overlap
        df_no_overlap = df[~(df['pdb'].isin(df_t))]
        cindex, p_corr, s_corr, mse, mae, rmse = get_metrics(df_no_overlap['actual'], df_no_overlap['pred'])
        results_without_overlap.append([cindex, p_corr[0], s_corr[0], mse, mae, rmse])

        if i==0 and plot:
            n_plots = int(compare_overlap)+1
            fig = plt.figure(figsize=(14,5*n_plots))
            axes = fig.subplots(n_plots,1)
            ax = axes[0] if compare_overlap else axes
            
            sns.histplot(df_no_overlap['actual'], kde=True, ax=ax, alpha=0.5, label='True pkd')
            sns.histplot(df_no_overlap['pred'], kde=True, ax=ax, alpha=0.5, label='Predicted pkd', color='orange')
            ax.set_title(f"{'Normalized 'if NORMALIZE else ''} pkd distribution")
            ax.legend()
            
            if compare_overlap:
                sns.histplot(df_no_overlap['actual'], kde=True, ax=axes[1], alpha=0.5, label='True pkd')
                sns.histplot(df_no_overlap['pred'], kde=True, ax=axes[1], alpha=0.5, label='Predicted pkd', color='orange')
                axes[1].set_title(f"{'Normalized 'if NORMALIZE else ''}  pkd distribution (no overlap)")
                axes[1].legend()

    if compare_overlap:
        return generate_markdown([results_with_overlap, results_without_overlap], names=['with overlap', 'without overlap'], 
                             cindex=True,verbose=verbose)
    # 'mean $\pm$ se'
    return generate_markdown([results_without_overlap], names=['mean predictive performance'], cindex=True, verbose=verbose)

def get_dpkd(df, pkd_col='pkd', normalize=False) -> np.ndarray:
    """ 
    2. Mutation impact analysis - Delta pkd given df containing wt and mutated proteins and their pkd values
    
    returns dpkd numpy array of shape (n,)
    """
    wt_df = df[df.index.str.contains("_wt")]
    mt_df = df[df.index.str.contains("_mt")]
    
    dpkd= []
    missing_wt = 0
    for m in mt_df.index:
        i_wt = m.split('_')[0] + '_wt'
        if i_wt not in wt_df.index:
            missing_wt += 1
            continue
        else:
            wt_pkd = wt_df.loc[i_wt][pkd_col]
            dpkd.append((wt_pkd - mt_df.loc[m][pkd_col]))

    dpkd = np.array(dpkd)
    
    if normalize:    
        # identify significance threshold seperately:
        mean_dpkd = np.mean(dpkd, axis=0)
        std_dpkd = np.std(dpkd, axis=0)
        
        dpkd = (dpkd - mean_dpkd) / std_dpkd # z-normalization        
    return dpkd

def fig_dpkd_dist(df, pkd_cols=['pred', 'actual'], normalize=False, show_plot=True, ax=None,) -> tuple[np.ndarray]:
    """ 
    2. Mutation impact analysis - Delta pkd distribution visualized overlayed pred and true distributions
    
    returns pred_dpkd, true_dpkd, and axes for the plot
    """
    assert len(pkd_cols) == 2
    
    pred_dpkd = get_dpkd(df, pkd_cols[0], normalize)
    true_dpkd = get_dpkd(df, pkd_cols[1], normalize)
        
    ax = sns.histplot(true_dpkd, kde=True, ax=ax, alpha=0.5, label='True Δpkd')#, color='blue')
    sns.histplot(pred_dpkd, kde=True, ax=ax, alpha=0.5, label='Predicted Δpkd', color='orange')
    ax.set_title(f"{'Normalized 'if normalize else ''}Δpkd distribution")
    ax.legend()
    if show_plot: plt.show()
    return pred_dpkd, true_dpkd, ax

def tbl_stratified_dpkd_metrics(
    MODEL = lambda i: f"results/model_media/test_set_pred/GVPLM_PDBbind{i}D_nomsaF_aflowE_128B_0.00022659LR_0.02414D_2000E_gvpLF_binaryLE_PLATINUM.csv",
    NORMALIZE=True,
    n_models=5,
    df_transform=get_mut_count, # transformation to apply to df before condition for grouping
    conditions = ["(n_mut == 1) | (n_mut == 0)", "(n_mut > 1) | (n_mut == 0)"],
    names = ['single mutation', '2+ mutations'],
    verbose=True,
    plot=False,
    **kwargs,
    ):
    """
    Generates markdown table for stratified results given some transform and conditions to performn on df for grouping.

    Args:
        `MODEL` (callable, optional): function to give path to model predictions. Defaults to lambdai:f"results/model_media/test_set_pred/GVPLM_PDBbind{i}D_nomsaF_aflowE_128B_0.00022659LR_0.02414D_2000E_gvpLF_binaryLE_PLATINUM.csv".
        `NORMALIZE` (bool, optional): Whether or not to scale predictions and truths before getting metrics. Defaults to True.
        `n_models` (int, optional): number of models to get data from (calls MODEL(i)). Defaults to 5.
        `df_transform` (callable, optional): returns a modified version of the predictions csv. Defaults to get_mut_count.
        `conditions`(list, optional): conditions to df.query() for grouping. Defaults to ["(n_mut == 1) | (n_mut == 0)", "(n_mut > 1) | (n_mut == 0)"].
        `names` (list, optional): names of each group. Defaults to ['single mutation', '2+ mutations'].
        `verbose` (bool, optional): Defaults to True.
        `plot` (bool, optional): Defaults to False.
        `**kwargs`: any additional args for the df_transform.

    Returns:
        pd.Dataframe: the table containing mean metrics and std for each group.
    """

    results = [[] for _ in range(len(conditions))]
    for i in range(n_models):
        df = pd.read_csv(MODEL(i), index_col=0).dropna()
        df['pdb'] = df['prot_id'].str.split('_').str[0]
        df = df_transform(df, **kwargs)
        
        if i == 0 and plot:
            fig = plt.figure(figsize=(14,5*len(conditions)))
            axes = fig.subplots(len(conditions),1)
        
        # must include 0 in both cases since they are the wildtype reference 
        for j, c in enumerate(conditions):
            grp = df.query(c)
            true_dpkd1 = get_dpkd(grp, 'actual', NORMALIZE)
            pred_dpkd1 = get_dpkd(grp, 'pred', NORMALIZE)
            
            if i==0 and plot:
                sns.histplot(true_dpkd1, kde=True, ax=axes[j], alpha=0.5, label='True Δpkd')
                sns.histplot(pred_dpkd1, kde=True, ax=axes[j], alpha=0.5, label='Predicted Δpkd', color='orange')
                axes[j].set_title(f"{'Normalized 'if NORMALIZE else ''}Δpkd distribution {names[j]}")
                axes[j].legend()

            _, p_corr, s_corr, mse, mae, rmse = get_metrics(true_dpkd1, pred_dpkd1)
            results[j].append([p_corr[0], s_corr[0], mse, mae, rmse])

    return generate_markdown(results, names=names, verbose=verbose)

def tbl_dpkd_metrics_overlap(
    MODEL = lambda i: f"results/model_media/test_set_pred/GVPLM_PDBbind{i}D_nomsaF_aflowE_128B_0.00022659LR_0.02414D_2000E_gvpLF_binaryLE_PLATINUM.csv",
    TRAIN_DATA_P = lambda set: f'{cfg.DATA_ROOT}/PDBbindDataset/nomsa_aflow_gvp_binary/{set}0/cleaned_XY.csv',
    NORMALIZE = True,
    verbose=True,
    plot=False,
    n_models=5,
    ):
    """
    2. Reports mean +- standard deviation with t-test significance of with/without overlap of training data from the 5-fold CV. 
    """
    
    df_t = pd.Index.append(pd.read_csv(TRAIN_DATA_P('train'), index_col=0).index, 
                        pd.read_csv(TRAIN_DATA_P('val'), index_col=0).index)
    df_t = df_t.str.upper()
    def get_in_train(df, training_set_df):
        df['in_train'] = df['pdb'].isin(training_set_df)
        return df

    conditions = ['(not in_train) | in_train', 'not in_train']
    names = ['with overlap', 'without overlap']

    return tbl_stratified_dpkd_metrics(MODEL, NORMALIZE, n_models, df_transform=get_in_train, 
                                       conditions=conditions, names=names, verbose=verbose, plot=plot, training_set_df=df_t)
    
def tbl_dpkd_metrics_n_mut(
    MODEL = lambda i: f"results/model_media/test_set_pred/GVPLM_PDBbind{i}D_nomsaF_aflowE_128B_0.00022659LR_0.02414D_2000E_gvpLF_binaryLE_PLATINUM.csv",
    NORMALIZE = True,
    n_models=5,
    conditions=[1,2],
    verbose=True,
    plot=False,
    ):
    """
    Conditions are the grouping for number of mutations with the last entry being all above that number:
    e.g.:
    conditions = [1,2]
    
    means we have two groups: single mutations and those with multiple (2+) mutations
    
    Any inbetween are considered as exact matches
    so conditions = [1,2,3]
    would mean 3 groups: exactly 1 mutation, exactly 2 mutations, and 3 or more mutations
    
    """
    names = []
    for i, c in enumerate(conditions):
        if i == len(conditions)-1:
            q = f"(n_mut >= {c}) | (n_mut == 0)"
            n = f"{c}+ mutations"
        else:
            q = f"(n_mut == {c}) | (n_mut == 0)"
            n = f"{c} mutations"
        
        conditions[i] = q
        names.append(n)

    return tbl_stratified_dpkd_metrics(MODEL, NORMALIZE, n_models, df_transform=get_mut_count,
                                       conditions=conditions, names=names, verbose=verbose, plot=plot)

def tbl_dpkd_metrics_in_binding(
    MODEL = lambda i: f"results/model_media/test_set_pred/GVPLM_PDBbind{i}D_nomsaF_aflowE_128B_0.00022659LR_0.02414D_2000E_gvpLF_binaryLE_PLATINUM.csv",
    RAW_PLT_CSV=f"{cfg.DATA_ROOT}/PlatinumDataset/raw/platinum_flat_file.csv",
    NORMALIZE = True,
    n_models=5,
    verbose=True,
    plot=False,
    ):
    """Generates a table comapring the metrics for mutations in the pocket and not in the pocket"""
    # add in_binding info to df
    def get_in_binding(df, dfr):
        """
        df is the predicted csv with index as <raw_idx>_wt (or *_mt) where raw_idx 
        corresponds to an index in dfr which contains the raw data for platinum including 
        ('mut.in_binding_site')
            - 0: wildtype rows
            - 1: in pocket
            - 2: outside of pocket
        """
        in_pocket = dfr[dfr['mut.in_binding_site'] == 'YES'].index   
        pclass = []
        for code in df.index:
            if '_wt' in code:
                pclass.append(0)
            elif int(code.split('_')[0]) in in_pocket:
                pclass.append(1)
            else:
                pclass.append(2)
                
        df['in_pocket'] = pclass
        return df

    conditions = ['(in_pocket == 0) | (in_pocket == 1)', '(in_pocket == 0) | (in_pocket == 2)']
    names = ['mutation in pocket', 'mutation NOT in pocket']

    dfr = pd.read_csv(RAW_PLT_CSV, index_col=0)
    
    dfp = pd.read_csv(MODEL(0), index_col=0)
    df = get_in_binding(dfp, dfr)
    if verbose: 
        cnts = df.in_pocket.value_counts()
        cnts.index = ['wt', 'pckt', 'not pckt']
        cnts.name = "counts"
        print(cnts.to_markdown(), end="\n\n")
    
    return tbl_stratified_dpkd_metrics(MODEL, NORMALIZE, n_models=n_models, df_transform=get_in_binding,
                                        conditions=conditions, names=names, verbose=verbose, plot=plot, dfr=dfr)

### 3. significant mutations as a classification problem
def fig_sig_mutations_conf_matrix(true_dpkd, pred_dpkd, std=2, verbose=True, plot=True, show_plot=False, ax=None):
    """For 3. significant mutation impact analysis"""
    dpkd = []
    # filter out nan vals
    for y,p in zip(true_dpkd, pred_dpkd):
        if not (np.isnan(y) or np.isnan(p)):
            dpkd.append((y,p))
    dpkd = np.array(dpkd)
    
    # identify significance threshold seperately:
    mean_dpkd = np.mean(dpkd, axis=0)
    std_dpkd = np.std(dpkd, axis=0)
    sig_thresh = mean_dpkd + std* std_dpkd # basically same effect as z-normalization (x-mean)/std

    # Mark observed mutations as significant or not
    sig_dpkd = abs(dpkd) > sig_thresh
    conf_matrix = confusion_matrix(sig_dpkd[:,0], sig_dpkd[:,1])

    if plot:
        disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, 
                                    display_labels=["significant", "not significant"])
        disp.plot(cmap=plt.cm.Blues, ax=ax)
        disp.ax_.set_title('Confusion Matrix for Mutation Significance')
        if show_plot: plt.show()

    # Calculate and print TPR and TNR
    tn, fp, fn, tp = conf_matrix.ravel()
    tpr = 0.0 if tp == 0 else tp / (tp + fn)
    tnr = 0.0 if tn == 0.0 else tn / (tn + fp)
    if verbose:
        print(f"True Positive Rate (TPR): {tpr:.2f}")
        print(f"True Negative Rate (TNR): {tnr:.2f}")
    return conf_matrix, tpr, tnr

def generate_roc_curve(true_dpkd, pred_dpkd, thres_range=(0,5), step=0.1):
    """3. significant mutation impact analysis"""
    
    # Define a range of standard deviations to use as thresholds
    std_values = np.arange(thres_range[0], thres_range[1], step)
    tprs = []
    fprs = []
    distances = []
    best_threshold = None
    min_distance = float('inf')  # Initialize with infinity

    # Store indices for std = 1.0 and std = 2.0
    index_std_1 = None
    index_std_2 = None

    for i, std in enumerate(std_values):
        # Use the confusion matrix function to get performance metrics at each threshold
        conf_matrix, tpr, tnr = fig_sig_mutations_conf_matrix(true_dpkd, pred_dpkd, std=std, 
                                                              verbose=False, plot=False, show_plot=False)
        fpr = 1 - tnr
        tprs.append(tpr)
        fprs.append(fpr)
        
        # Calculate the Euclidean distance from the top-left corner (0,1)
        distance = np.sqrt((0 - fpr) ** 2 + (1 - tpr) ** 2)
        distances.append(distance)
        if distance < min_distance:
            min_distance = distance
            best_threshold = std
            best_tpr = tpr
            best_fpr = fpr
        
        # Check if the current std is 1.0 or 2.0
        if np.isclose(std, 1.0, atol=0.05):
            index_std_1 = i
        elif np.isclose(std, 2.0, atol=0.05):
            index_std_2 = i

    # Plot the ROC curve
    plt.figure(figsize=(16, 12))
    plt.plot(fprs, tprs, marker='o', linestyle='-', color='b', label='ROC Curve')
    plt.plot([0, 1], [0, 1], 'k--', label='No Skill Line')  # diagonal line for reference
    
    # Highlight the best point
    plt.scatter([best_fpr], [best_tpr], color='red', s=150, edgecolors='k', label=f'Best Threshold (STD={best_threshold:.1f})')
    # Highlight specific std points
    plt.scatter([fprs[index_std_1]], [tprs[index_std_1]], color='green', s=150, edgecolors='k', label='STD=1.0')
    plt.scatter([fprs[index_std_2]], [tprs[index_std_2]], color='purple', s=150, edgecolors='k', label='STD=2.0')
    
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.grid(True)
    plt.legend()
    plt.show()

    print(f"Best threshold: {best_threshold} with minimum distance: {min_distance}")
    return std_values, tprs, fprs, best_threshold

if __name__ == '__main__':
    # %%
    import matplotlib.pyplot as plt
    import seaborn as sns 

    from src.utils import config as cfg
    from src.analysis.figures import (prepare_df, fig1_pro_overlap, 
                                        fig2_pro_feat, fig3_edge_feat, 
                                        fig4_pro_feat_violin, fig5_edge_feat_violin)

    # %%
    df = prepare_df(csv_p=cfg.MODEL_STATS_CSV, old_csv_p="results/model_media/old_model_stats.csv")

    #%% display figures
    verbose = False

    #%% dataset comparisons
    for col in ['cindex', 'pearson']:
        fig1_pro_overlap(df, sel_col=col, verbose=verbose, show=False)
        plt.savefig(f"results/figures/fig1_pro_overlap_{col}.png", dpi=300, bbox_inches='tight')
        plt.clf()
        fig2_pro_feat(df, sel_col=col, verbose=verbose, context='paper', add_labels=False, show=False)
        plt.savefig(f"results/figures/fig2_pro_feat_{col}.png", dpi=300, bbox_inches='tight')
        plt.clf()
        fig3_edge_feat(df, sel_col=col, exclude=['af2-anm'], verbose=verbose, context='paper', add_labels=False, show=False)
        plt.savefig(f"results/figures/fig3_edge_feat_{col}.png", dpi=300, bbox_inches='tight')
        plt.clf()

    # %% Davis violin plots
    sns.set(style="darkgrid")
    for dataset in ['davis', 'kiba', 'PDBbind']:
        for col in ['cindex', 'mse']:
            fig4_pro_feat_violin(df, sel_col=col, sel_dataset=dataset, verbose=verbose, show=False)
            plt.savefig(f"results/figures/fig4_pro_feat_violin_{dataset}_{col}.png", dpi=300, bbox_inches='tight')
            plt.clf()
            fig5_edge_feat_violin(df, sel_col=col, sel_dataset=dataset, exclude=['af2-anm'], verbose=verbose, show=False)
            plt.savefig(f"results/figures/fig5_edge_feat_violin_{dataset}_{col}.png", dpi=300, bbox_inches='tight')
            plt.clf()
            
            
    # %% Combined violin plots
    fig_combined(df, datasets=['PDBbind','davis', 'kiba'], metrics=['cindex', 'mse'], fig_callable=fig4_pro_feat_violin)
    plt.savefig(f'results/figures/fig_combined_proViolin_CI-MSE.png', dpi=300, bbox_inches='tight')
    plt.clf()
    fig_combined(df, datasets=['PDBbind','davis', 'kiba'], metrics=['cindex', 'mse', 'pearson'], fig_callable=fig4_pro_feat_violin)
    plt.savefig(f'results/figures/fig_combined_proViolin_CI-MSE-Pearson.png', dpi=300, bbox_inches='tight')
    plt.clf()

    fig_combined(df, datasets=['PDBbind','davis', 'kiba'], metrics=['cindex', 'mse'], fig_callable=fig5_edge_feat_violin)
    plt.savefig(f"results/figures/fig_combined_edgeViolin_CI-MSE.png", dpi=300, bbox_inches='tight')
    plt.clf()
    fig_combined(df, datasets=['PDBbind','davis', 'kiba'], metrics=['cindex', 'mse', 'pearson'], fig_callable=fig5_edge_feat_violin)
    plt.savefig(f"results/figures/fig_combined_edgeViolin_CI-MSE-Pearson.png", dpi=300, bbox_inches='tight')
    plt.clf()


    #%% dataset comparisons
    plot_df = fig1_pro_overlap(df, sel_col='mse', verbose=verbose, show=False)

    # performance drop values:
    grp = plot_df.groupby(['data', 'overlap']).mean()
    grp[grp.index.get_level_values(1)].values - grp[~grp.index.get_level_values(1)].values
    
    #%%
    ########################################################################
    ########################## VIOLIN PLOTTING #############################
    ########################################################################
    import logging
    from matplotlib import pyplot as plt

    from src.analysis.figures import prepare_df, fig_combined, custom_fig

    dft = prepare_df('./results/v115/model_media/model_stats.csv')
    dfv = prepare_df('./results/v115/model_media/model_stats_val.csv')

    models = {
        'DG': ('nomsa', 'binary', 'original', 'binary'),
        # 'esm': ('ESM', 'binary', 'original', 'binary'), # esm model
        'aflow': ('nomsa', 'aflow', 'original', 'binary'),
        # 'gvpP': ('gvp', 'binary', 'original', 'binary'),
        # 'gvpL': ('nomsa', 'binary', 'gvp', 'binary'),
        # 'aflow_ring3': ('nomsa', 'aflow_ring3', 'original', 'binary'),
        # 'gvpL_aflow': ('nomsa', 'aflow', 'gvp', 'binary'),
        # 'gvpL_aflow_rng3': ('nomsa', 'aflow_ring3', 'gvp', 'binary'),
        #GVPL_ESMM_davis3D_nomsaF_aflowE_48B_0.00010636872718329864LR_0.23282479481785903D_2000E_gvpLF_binaryLE
        # 'gvpl_esm_aflow': ('ESM', 'aflow', 'gvp', 'binary'),
    }

    fig, axes = fig_combined(dft, datasets=['davis'], fig_callable=custom_fig,
                models=models, metrics=['cindex', 'mse'],
                fig_scale=(10,5), add_stats=True, title_postfix=" test set performance", box=True, fold_labels=True)
    plt.xticks(rotation=45)

    fig, axes = fig_combined(dfv, datasets=['davis'], fig_callable=custom_fig,
                models=models, metrics=['cindex', 'mse'],
                fig_scale=(10,5), add_stats=True, title_postfix=" validation set performance", box=True, fold_labels=True)
    plt.xticks(rotation=45)
