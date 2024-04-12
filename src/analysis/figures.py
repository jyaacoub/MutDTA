from collections import Counter, OrderedDict
import logging
import os, pickle, json

import pandas as pd
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from statannotations.Annotator import Annotator

from src.utils import config as cfg
from src.utils.loader import Loader

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
                  show=False, **kwargs):    
    # Create subplots with datasets as columns and metrics as rows
    fig, axes = plt.subplots(len(metrics), len(datasets), 
                             figsize=(fig_scale[0]*len(datasets), 
                                      fig_scale[1]*len(metrics)))
    for i, dataset in enumerate(datasets):
        for j, metric in enumerate(metrics):
            # Set current subplot
            if len(datasets) == 1 or len(metrics) == 1:
                ax = axes[j] if len(datasets) == 1 else axes[i]
            else:
                ax = axes[j, i]

            fig_callable(df, sel_col=metric, sel_dataset=dataset, show=False, 
                         ax=ax, **kwargs)
                        
            # Add titles only to the top row and left column
            if j == 0:
                ax.set_title(f'{dataset}')
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
                   verbose=False, show=False, add_stats=True, ax=None):
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
    plot_data = OrderedDict()
    for model, feat in models.items():
        plot_data[model] = filtered_df[matched(filtered_df, feat)][sel_col]
        if len(plot_data[model]) != 5:
            logging.warning(f'Expected 5 results for {model}, got {len(plot_data[model])}')

    # plot violin plot with annotations
    vals = list(plot_data.values())
    ax = sns.violinplot(data=vals, ax=ax)
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

def prepare_df(csv_p:str=cfg.MODEL_STATS_CSV, old_csv_p:str=None) -> pd.DataFrame:
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
    
    df['lr'] = df['run'].str.extract(r'_(\d+\.?\d*)LR_', expand=False).astype(float)
    df['dropout'] = df['run'].str.extract(r'_(\d+\.?\d*)D_', expand=False).astype(float)
    
    # ESM models
    df.loc[df['run'].str.contains('EDM') & df['run'].str.contains('nomsaF'), 'feat'] = 'ESM'
    df.loc[df['run'].str.contains('EDAM'), 'feat'] += '-ESM'
    df.loc[df['run'].str.contains('EDIM') & df['run'].str.contains('nomsaF'), 'feat'] = 'ESM'
    df.loc[df['run'].str.contains('EDAIM'), 'feat'] += '-ESM'

    # ChemGPT models
    df.loc[df['run'].str.contains('CDM'), 'lig_feat'] = 'ChemGPT'
    #TODO: account for ChemGPT with additional features

    df['overlap'] = df['run'].str.contains('overlap')
    
    return df


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
