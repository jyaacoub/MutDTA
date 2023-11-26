import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from statannotations.Annotator import Annotator
from src.utils import config as cfg
from src.utils.loader import Loader

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
                     & (df['fold'] != '') & (df['lig_feat'].isna())]
    
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
                     & (df['fold'] != '') & (df['lig_feat'].isna())]
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
                         show=False, add_labels=True, add_stats=True, ax=None):
    # Extract relevant data
    filtered_df = df[(df['edge'] == 'binary') & (~df['overlap']) & (df['lig_feat'].isna())]
    
    # show all with fold info
    filtered_df = filtered_df[(filtered_df['data'] == sel_dataset) & (filtered_df['fold'] != '')]
    nomsa = filtered_df[(filtered_df['feat'] == 'nomsa')][sel_col]
    msa = filtered_df[(filtered_df['feat'] == 'msa')][sel_col]
    shannon = filtered_df[(filtered_df['feat'] == 'shannon')][sel_col]
    esm = filtered_df[(filtered_df['feat'] == 'ESM')][sel_col]

    # printing length of each feature
    if verbose:
        print(f'nomsa: {len(nomsa)}')
        print(f'msa: {len(msa)}')
        print(f'shannon: {len(shannon)}')
        print(f'esm: {len(esm)}')


    # Get values for each node feature
    plot_data = [nomsa, msa, shannon, esm]
    ax = sns.violinplot(data=plot_data, ax=ax)
    ax.set_xticklabels(['nomsa', 'msa', 'shannon', 'esm'])
    ax.set_ylabel(sel_col)
    ax.set_xlabel('Features')
    ax.set_title(f'Feature {sel_col} for {sel_dataset}')
    
    # Annotation for stats
    if add_stats:
        pairs = [(0,1), (0,2), (1,2)]
        if len(esm) > 0: 
            pairs += [(0,3),(1,3), (2,3)] # add esm pairs if esm is not empty
        annotator = Annotator(ax, pairs, data=plot_data, verbose=verbose)
        annotator.configure(test='Mann-Whitney', text_format='star', loc='inside', 
                            hide_non_significant=not verbose)
        annotator.apply_and_annotate()

    if show:
        plt.show()
        
    return nomsa, msa, shannon, esm

# Figure 5: violin plot with error bars for Cross-validation results to show significance among edge feats
def fig5_edge_feat_violin(df, sel_dataset='davis', verbose=False, sel_col='cindex', exclude=[],
                            show=False, add_labels=True, add_stats=True, ax=None):
    filtered_df = df[(df['feat'] == 'nomsa') & (~df['overlap']) & (df['lig_feat'].isna())]
    filtered_df = filtered_df[(filtered_df['data'] == sel_dataset) & (filtered_df['fold'] != '')]

    filtered_df.sort_values(by=['edge'], inplace=True)

    # Get values for each edge type
    binary = filtered_df[filtered_df['edge'] == 'binary'][sel_col]
    simple = filtered_df[filtered_df['edge'] == 'simple'][sel_col]
    anm = filtered_df[filtered_df['edge'] == 'anm'][sel_col]
    af2 = filtered_df[filtered_df['edge'] == 'af2'][sel_col]

    # plot violin plot with annotations
    plot_data = [binary, simple, anm, af2]
    ax = sns.violinplot(data=plot_data, ax=ax)
    ax.set_xticklabels(['binary', 'simple', 'anm', 'af2'])
    ax.set_ylabel(sel_col)
    ax.set_xlabel('Edge type')
    ax.set_title(f'Edge type {sel_col} for {sel_dataset}')

    if add_stats:
        pairs = [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]
        annotator = Annotator(ax, pairs, data=plot_data, verbose=verbose)
        annotator.configure(test='Mann-Whitney', text_format='star', loc='inside', 
                            hide_non_significant=not verbose)
        annotator.apply_and_annotate()
        
    if show:
        plt.show()
        
    return binary, simple, anm, af2

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
                  fig_callable=fig4_pro_feat_violin,
                  show=False, **kwargs):    
    # Create subplots with datasets as columns and cols as rows
    fig, axes = plt.subplots(len(metrics), len(datasets), 
                             figsize=(5*len(datasets), 4*len(metrics)))
    for i, dataset in enumerate(datasets):
        for j, metric in enumerate(metrics):
            # Set current subplot
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
    df['feat'] = df['run'].str.extract(r'_(nomsa|msa|shannon)F_', expand=False)
    df['edge'] = df['run'].str.extract(r'_(binary|simple|anm|af2|af2-anm)E_', expand=False)
    df['ddp'] = df['run'].str.contains('DDP-')
    df['improved'] = df['run'].str.contains('IM_') # postfix of model name will include I if "improved"
    df['batch_size'] = df['run'].str.extract(r'_(\d+)B_', expand=False)
    
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
    from src.data_analysis.figures import (prepare_df, fig1_pro_overlap, 
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

    #%% dataset comparisons
    plot_df = fig1_pro_overlap(df, sel_col='mse', verbose=verbose, show=False)

    # performance drop values:
    grp = plot_df.groupby(['data', 'overlap']).mean()
    grp[grp.index.get_level_values(1)].values - grp[~grp.index.get_level_values(1)].values
