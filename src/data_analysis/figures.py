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
def fig2_pro_feat(df, verbose=False, sel_col='cindex', show=True):
    # Extract relevant data
    filtered_df = df[(df['edge'] == 'binary') & (~df['overlap'])]
    
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
        if sel_col == 'cindex':
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

    # Melt the DataFrame for Seaborn barplot
    melted_data = pd.melt(plot_data, id_vars=['Dataset'], var_name='Node feature', 
                          value_name=sel_col)

    # Create a bar plot using Seaborn
    plt.figure(figsize=(14, 7))
    sns.set(style="darkgrid")
    sns.set_context('poster')
    ax = sns.barplot(x='Dataset', y=sel_col, hue='Node feature', 
                     data=melted_data, palette='deep')
    # Set the title
    ax.set_title(f'Node feature performance ({"concordance index" if sel_col == "cindex" else "MSE"})')
    
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
def fig3_edge_feat(df, verbose=False, sel_col='cindex', show_simple=True, show=True):
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

            if sel_col == 'cindex':
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
    })

    # Melt the DataFrame for Seaborn barplot
    melted_data = pd.melt(plot_data, id_vars=['Dataset'], var_name='Edge type', 
                            value_name=sel_col)

    # Create a bar plot using Seaborn
    plt.figure(figsize=(14, 7))
    sns.set(style="darkgrid")
    sns.set_context('poster')
    ax = sns.barplot(x='Dataset', y=sel_col, hue='Edge type', 
                        data=melted_data, palette='deep')
    # Set the title
    ax.set_title(f'Edge type performance ({"concordance index" if sel_col == "cindex" else "MSE"})')

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


if __name__ == '__main__':
    csv = 'results/model_media/model_stats.csv'
    
    df = pd.read_csv(csv)
    df = pd.concat([df, pd.read_csv('results/model_media/old_model_stats.csv')]) # concat with old model results since we get the max value anyways...

    # create data, feat, and overlap columns for easier filtering.
    df['data'] = df['run'].str.extract(r'_(davis|kiba|PDBbind)', expand=False)
    df['feat'] = df['run'].str.extract(r'_(nomsa|msa|shannon)F_', expand=False)
    df['edge'] = df['run'].str.extract(r'_(binary|simple|anm|af2|af2-anm)E_', expand=False)
    df['ddp'] = df['run'].str.contains('DDP-')
    df['improved'] = df['run'].str.contains('IM_') # trail of model name will include I if "improved"
    df['batch_size'] = df['run'].str.extract(r'_(\d+)B_', expand=False)
    
    df.loc[df['run'].str.contains('EDM') & df['run'].str.contains('nomsaF'), 'feat'] = 'ESM'
    df.loc[df['run'].str.contains('EDAM'), 'feat'] += '-ESM'
    df.loc[df['run'].str.contains('EDIM') & df['run'].str.contains('nomsaF'), 'feat'] = 'ESM'
    df.loc[df['run'].str.contains('EDAIM'), 'feat'] += '-ESM'

    df['overlap'] = df['run'].str.contains('overlap')

    df[['run', 'data', 'feat', 'edge', 'batch_size', 'overlap']]

    fig1_pro_overlap(df, show=True)
    fig2_pro_feat(df, show=True)
    fig3_edge_feat(df, show=True)