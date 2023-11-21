import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from statannotations.Annotator import Annotator

# Figure 1 - Protein overlap cindex difference (nomsa)
def fig1_pro_overlap(df, sel_col='cindex', verbose=False, show=True):
    grouped_df = df[(df['feat'] == 'nomsa') 
                    & (df['batch_size'] == '64') 
                    & (df['edge'] == 'binary')
                    & (~df['ddp'])              
                    & (~df['improved'])].groupby(['data'])
    
    # each group is a dataset with 2 bars (overlap and no overlap)
    for group_name, group_data in grouped_df:
        if verbose: print(f"\nGroup Name: {group_name}")
        if verbose: print(group_data[['cindex', 'mse', 'overlap']])

    # these groups are spaced by the data type, physically grouping bars of the same dataset together.
    # Initialize lists to store cindex values for each dataset type
    t_overlap = []
    f_overlap = []
    dataset_types = []

    for dataset, group in grouped_df:
        if verbose: print('')
        if verbose: print(group[['cindex', 'mse', 'overlap', 'data']])
        
        # overlap
        col = group[group['overlap']][sel_col]
        t_overlap_val = col.max() if sel_col == 'cindex' else col.min()
        if np.isnan(t_overlap_val):
            t_overlap_val = 0
        t_overlap.append(t_overlap_val)
        
        # no overlap
        col = group[~group['overlap']][sel_col]
        f_overlap_val = col.max() if sel_col == 'cindex' else col.min()
        if np.isnan(f_overlap_val):
            f_overlap_val = 0
        f_overlap.append(f_overlap_val)
        dataset_types.append(dataset[0])

    # Create an array of x positions for the bars
    x = np.arange(len(dataset_types))

    # Set the width of the bars
    width = 0.35

    # Create a bar plot with two bars for each dataset type
    fig, ax = plt.subplots()
    bar2 = ax.bar(x - width/2, t_overlap, width, label='With Overlap')
    bar1 = ax.bar(x + width/2, f_overlap, width, label='No Overlap')

    # Set the x-axis labels
    ax.set_xticks(x)
    ax.set_xticklabels(dataset_types)

    # Set the y-axis label
    ax.set_ylabel(sel_col)
    if sel_col == 'cindex':
        ax.set_ylim([0.5, 1]) # 0.5 is the worst cindex value

    # Set the title and legend
    ax.set_title(f'Protein Overlap {sel_col} Difference (nomsa)')
    ax.legend()

    # Show the plot
    if show: plt.show()

# Figure 2 - node feature cindex difference
# Features -> nomsa, msa, shannon, and esm
def fig2_pro_feat(df, verbose=False, sel_col='cindex', exclude=[], show=True, add_labels=True):
    # Extract relevant data
    filtered_df = df[(df['edge'] == 'binary') & (~df['overlap']) & (df['lig_feat'].isna())] # NOTE 
    
    # Initialize lists to store cindex values for each dataset type
    nomsa = []
    msa = []
    shannon = []
    esm = []
    dataset_types = []

    for dataset, group in filtered_df.groupby('data'):
        if verbose: print(f"\nGroup Name: {dataset}")
        if verbose: print(group[['cindex', 'mse', 'feat']])
        
        # Extract max or min values based on sel_col
        if sel_col in ['cindex', 'pearson', 'spearman']:
            nomsa_v = group[group['feat'] == 'nomsa'][sel_col].max()
            msa_v = group[group['feat'] == 'msa'][sel_col].max()
            shannon_v = group[group['feat'] == 'shannon'][sel_col].max()
            ESM_v = group[group['feat'] == 'ESM'][sel_col].max()
        else:
            nomsa_v = group[group['feat'] == 'nomsa'][sel_col].min()
            msa_v = group[group['feat'] == 'msa'][sel_col].min()
            shannon_v = group[group['feat'] == 'shannon'][sel_col].min()
            ESM_v = group[group['feat'] == 'ESM'][sel_col].min()
        
        # Append values or 0 if NaN
        nomsa.append(nomsa_v if not np.isnan(nomsa_v) else 0)
        msa.append(msa_v if not np.isnan(msa_v) else 0)
        shannon.append(shannon_v if not np.isnan(shannon_v) else 0)
        esm.append(ESM_v if not np.isnan(ESM_v) else 0)
        dataset_types.append(dataset)

    # Create a DataFrame for plotting
    plot_data = pd.DataFrame({
        'Dataset': dataset_types,
        'nomsa': nomsa,
        'msa': msa,
        'shannon': shannon,
        'esm': esm
    })
    for c in exclude:
        plot_data.drop(c, axis=1, inplace=True)

    # Melt the DataFrame for Seaborn barplot
    melted_data = pd.melt(plot_data, id_vars=['Dataset'], var_name='Node feature', 
                          value_name=sel_col)

    # Create a bar plot using Seaborn
    plt.figure(figsize=(14, 7))
    sns.set(style="darkgrid")
    sns.set_context('poster')
    ax = sns.barplot(x='Dataset', y=sel_col, hue='Node feature', 
                     data=melted_data, palette='deep')
    if add_labels:
        for i in ax.containers: 
            ax.bar_label(i, fmt='%.3f', fontsize=13)
            
    # Set the title
    ax.set_title(f'Node feature performance ({"concordance index" if sel_col == "cindex" else sel_col})')
    
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles, labels=labels, loc='upper right')
    
    # Set the y-axis label and limit
    ax.set_ylabel(sel_col)
    if sel_col == 'cindex':
        ax.set_ylim([0.5, 1])  # 0.5 is the worst cindex value
    
    # Add statistical annotations
    pairs=[("PDBbind", "kiba"), ("PDBbind", "davis"), ("davis", "kiba")]
    annotator = Annotator(ax, pairs, data=df, x='data', y='cindex', order=['PDBbind', 'davis', 'kiba'])
    annotator.configure(test='Mann-Whitney', text_format='star', loc='inside', hide_non_significant=True,
                        line_height=0.005, verbose=verbose)
    annotator.apply_and_annotate()
    # Show the plot
    if show:
        plt.show()
        
    # reset stylesheet back to defaults
    mpl.rcParams.update(mpl.rcParamsDefault)

# Figure 3 - Edge type cindex difference
# Edges -> binary, simple, anm, af2
def fig3_edge_feat(df, verbose=False, sel_col='cindex', exclude=[], show=True, add_labels=True):
    # comparing nomsa, msa, shannon, and esm
    # group by data type
    
    # this will capture multiple models per dataset (different LR, batch size, etc)
    #   Taking the max cindex value for each dataset will give us the best model for each dataset
    filtered_df = df[(df['feat'] == 'nomsa') & (~df['overlap'])]
    
    # these groups are spaced by the data type, physically grouping bars of the same dataset together.
    # Initialize lists to store cindex values for each dataset type
    binary = []
    simple = []
    anm = []
    af2 = []
    af2_anm = []
    dataset_types = []
    
    for dataset, group in filtered_df.groupby('data'):
        if verbose: print('')
        if verbose: print(group[['cindex', 'mse', 'overlap', 'data']])
        
        value_dict = {}

        for edge_value in ['binary', 'simple', 'anm', 'af2', 'af2-anm']:
            filtered_group = group[group['edge'] == edge_value]

            if sel_col == ['cindex', 'pearson', 'spearman']:
                value = filtered_group[sel_col].max()
            else:
                value = filtered_group[sel_col].min()

            value_dict[edge_value] = value if not np.isnan(value) else 0

        binary.append(value_dict['binary'])
        simple.append(value_dict['simple'])
        anm.append(value_dict['anm'])
        af2.append(value_dict['af2'])
        af2_anm.append(value_dict['af2-anm'])
        dataset_types.append(dataset)
    
    # Create a DataFrame for plotting
    plot_data = pd.DataFrame({
        'Dataset': dataset_types,
        'binary': binary,
        'simple': simple,
        'anm': anm,
        'af2': af2,
        'af2-anm': af2_anm
    })
    for c in exclude:
        plot_data.drop(c, axis=1, inplace=True)

    # Melt the DataFrame for Seaborn barplot
    melted_data = pd.melt(plot_data, id_vars=['Dataset'], var_name='Edge type', 
                            value_name=sel_col)

    # Create a bar plot using Seaborn
    plt.figure(figsize=(14, 7))
    sns.set(style="darkgrid")
    sns.set_context('poster')
    ax = sns.barplot(x='Dataset', y=sel_col, hue='Edge type', 
                        data=melted_data, palette='deep')
    if add_labels:
        for i in ax.containers: 
            ax.bar_label(i, fmt='%.3f', fontsize=13)
    # Set the title
    ax.set_title(f'Edge type performance ({"concordance index" if sel_col == "cindex" else sel_col})')

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles, labels=labels, loc='upper right')

    # Set the y-axis label and limit
    ax.set_ylabel(sel_col)
    if sel_col == 'cindex':
        ax.set_ylim([0.5, 1])  # 0.5 is the worst cindex value

    # Add statistical annotations
    pairs=[("PDBbind", "kiba"), ("PDBbind", "davis"), ("davis", "kiba")]
    annotator = Annotator(ax, pairs, data=df, x='data', y='cindex', order=['PDBbind', 'davis', 'kiba'])
    annotator.configure(test='Mann-Whitney', text_format='star', loc='inside', hide_non_significant=True,
                        line_height=0.005, verbose=False)
    annotator.apply_and_annotate()
    # Show the plot
    if show:
        plt.show()
        
    # reset stylesheet back to defaults
    mpl.rcParams.update(mpl.rcParamsDefault)    

# Figure 4: violin plot with error bars for Cross-validation results to show significance among pro feats
def fig4_pro_feat_violin(df, sel_dataset='davis', verbose=False, sel_col='cindex', exclude=[], 
                         show=True, add_labels=True, add_stats=True):
    # Extract relevant data
    filtered_df = df[(df['edge'] == 'binary') & (~df['overlap']) & (df['lig_feat'].isna())]

    # show all with fold info
    filtered_df = filtered_df[(filtered_df['data'] == sel_dataset) & (filtered_df['fold'] != '')]
    nomsa = filtered_df[(filtered_df['feat'] == 'nomsa')][sel_col]
    msa = filtered_df[(filtered_df['feat'] == 'msa')][sel_col]
    shannon = filtered_df[(filtered_df['feat'] == 'shannon')][sel_col]
    esm = filtered_df[(filtered_df['feat'] == 'ESM')][sel_col]

    # Get values for each node feature
    ax = sns.violinplot(data=[nomsa, msa, shannon, esm])
    ax.set_xticklabels(['nomsa', 'msa', 'shannon', 'esm'])
    ax.set_ylabel(sel_col)
    ax.set_xlabel('Features')
    ax.set_title(f'Feature {sel_col} for {sel_dataset}')
    
    # Annotation for stats
    if add_stats:
        pairs=[('nomsa', 'msa'), ('nomsa', 'shannon'), ('msa', 'shannon')]
        if len(esm) > 0: 
            pairs += [('ESM', 'nomsa'), ('ESM', 'msa'), ('ESM', 'shannon')]
        annotator = Annotator(ax, pairs, data=filtered_df, x='feat', y=sel_col, verbose=verbose)
        annotator.configure(test='Mann-Whitney', text_format='star', loc='inside', 
                            hide_non_significant=not verbose)
        annotator.apply_and_annotate()

    if show:
        plt.show()

# Figure 5: violin plot with error bars for Cross-validation results to show significance among edge feats
def fig5_edge_feat_violin(df, sel_dataset='davis', verbose=False, sel_col='cindex', exclude=[],
                            show=True, add_labels=True, add_stats=True):
    filtered_df = df[(df['feat'] == 'nomsa') & (~df['overlap']) & (df['lig_feat'].isna())]
    filtered_df = filtered_df[(filtered_df['data'] == sel_dataset) & (filtered_df['fold'] != '')]

    filtered_df.sort_values(by=['edge'], inplace=True)

    # Get values for each edge type
    binary = filtered_df[filtered_df['edge'] == 'binary'][sel_col]
    simple = filtered_df[filtered_df['edge'] == 'simple'][sel_col]
    anm = filtered_df[filtered_df['edge'] == 'anm'][sel_col]
    af2 = filtered_df[filtered_df['edge'] == 'af2'][sel_col]

    # plot violin plot with annotations
    ax = sns.violinplot(data=[binary, simple, anm, af2])
    ax.set_xticklabels(['binary', 'simple', 'anm', 'af2'])
    ax.set_ylabel(sel_col)
    ax.set_xlabel('Edge type')
    ax.set_title(f'Edge type {sel_col} for {sel_dataset}')

    if add_stats:
        pairs = [('binary', 'simple'), ('binary', 'anm'), ('binary', 'af2'), 
                ('simple', 'anm'), ('simple', 'af2'), ('anm', 'af2')]
        annotator = Annotator(ax, pairs, data=filtered_df, x='edge', y=sel_col, verbose=verbose)
        annotator.configure(test='Mann-Whitney', text_format='star', loc='inside', hide_non_significant=True)
        annotator.apply_and_annotate()
        
    if show:
        plt.show()



def prepare_df(csv_p:str, old_csv_p='results/model_media/old_model_stats.csv') -> pd.DataFrame:
    df = pd.read_csv(csv_p)
    
    if os.path.isfile(old_csv_p):
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
    df = prepare_df('results/model_media/model_stats.csv')

    df[['run', 'data', 'feat', 'edge', 'batch_size', 'overlap']]

    fig1_pro_overlap(df, show=True)
    fig2_pro_feat(df, show=True)
    fig3_edge_feat(df, show=True)