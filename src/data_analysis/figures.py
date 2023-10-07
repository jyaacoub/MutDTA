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

# Figure 2 - Feature type cindex difference
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

    for dataset_name, dataset_data in filtered_df.groupby('data'):
        if verbose:
            print(f"\nGroup Name: {dataset_name}")
        if verbose:
            print(dataset_data[['cindex', 'mse', 'feat']])
        
        # Extract max or min values based on sel_col
        if sel_col == 'cindex':
            nomsa_v = dataset_data[dataset_data['feat'] == 'nomsa']['cindex'].max()
            msa_v = dataset_data[dataset_data['feat'] == 'msa']['cindex'].max()
            shannon_v = dataset_data[dataset_data['feat'] == 'shannon']['cindex'].max()
            ESM_v = dataset_data[dataset_data['feat'] == 'ESM']['cindex'].max()
        else:
            nomsa_v = dataset_data[dataset_data['feat'] == 'nomsa']['cindex'].min()
            msa_v = dataset_data[dataset_data['feat'] == 'msa']['cindex'].min()
            shannon_v = dataset_data[dataset_data['feat'] == 'shannon']['cindex'].min()
            ESM_v = dataset_data[dataset_data['feat'] == 'ESM']['cindex'].min()
        
        # Append values or 0 if NaN
        nomsa.append(nomsa_v if not np.isnan(nomsa_v) else 0)
        msa.append(msa_v if not np.isnan(msa_v) else 0)
        shannon.append(shannon_v if not np.isnan(shannon_v) else 0)
        esm.append(ESM_v if not np.isnan(ESM_v) else 0)
        dataset_types.append(dataset_name)

    # Create a DataFrame for plotting
    plot_data = pd.DataFrame({
        'Dataset': dataset_types,
        'nomsa': nomsa,
        'msa': msa,
        'shannon': shannon,
        'esm': esm
    })

    # Melt the DataFrame for Seaborn barplot
    melted_data = pd.melt(plot_data, id_vars=['Dataset'], var_name='Feature Type', 
                          value_name=sel_col)

    # Create a bar plot using Seaborn
    plt.figure(figsize=(14, 7))
    sns.set(style="darkgrid")
    sns.set_context('poster')
    ax = sns.barplot(x='Dataset', y=sel_col, hue='Feature Type', 
                     data=melted_data, palette='deep')
    # Set the title
    ax.set_title(f'Feature type performance ({"concordance index" if sel_col == "cindex" else "MSE"})')
    
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

# Figure 3 - Edge type cindex difference
# Edges -> binary, simple, anm, af2
def fig3_edge_feat(df, verbose=False, sel_col='cindex', show=True):
    # comparing nomsa, msa, shannon, and esm
    # group by data type
    
    # this will capture multiple models per dataset (different LR, batch size, etc)
    #   Taking the max cindex value for each dataset will give us the best model for each dataset
    grouped_df = df[(df['feat'] == 'nomsa')
                    & (~df['overlap'])].groupby(['data'])
    
    # each group is a dataset with 4 bars (nomsa, msa, shannon, esm)
    for group_name, group_data in grouped_df:
        if verbose: print(f"\nGroup Name: {group_name}")
        if verbose: print(group_data[['cindex', 'mse', 'edge']])

    # these groups are spaced by the data type, physically grouping bars of the same dataset together.
    # Initialize lists to store cindex values for each dataset type
    binary = []
    simple = []
    anm = []
    af2 = []
    dataset_types = []
    
    for dataset, group in grouped_df:
        if verbose: print('')
        if verbose: print(group[['cindex', 'mse', 'overlap', 'data']])
        
        if sel_col == 'cindex':
            binary_v = group[group['edge'] == 'binary'][sel_col].max()
            simple_v = group[group['edge'] == 'simple'][sel_col].max()
            anm_v = group[group['edge'] == 'anm'][sel_col].max()
            af2_v = group[group['edge'] == 'af2'][sel_col].max()
        else:
            binary_v = group[group['edge'] == 'binary'][sel_col].min()
            simple_v = group[group['edge'] == 'simple'][sel_col].min()
            anm_v = group[group['edge'] == 'anm'][sel_col].min()
            af2_v = group[group['edge'] == 'af2'][sel_col].min()
        
        # appending if not nan else 0
        binary.append( binary_v if not np.isnan(binary_v) else 0)
        simple.append( simple_v if not np.isnan(simple_v) else 0)
        anm.append(anm_v if not np.isnan(anm_v) else 0)
        af2.append(af2_v if not np.isnan(af2_v) else 0)
        dataset_types.append(dataset[0])
        
    # Create an array of x positions for the bars
    x = np.arange(len(dataset_types))
    
    # Set the width of the bars
    width = 0.2
    
    # Create a bar plot with 4 bars for each dataset type
    fig, ax = plt.subplots()
    bar1 = ax.bar(x - width, binary, width, label='binary')
    bar2 = ax.bar(x, simple, width, label='simple')
    bar3 = ax.bar(x + width,   anm, width, label='anm')
    bar4 = ax.bar(x + width*2, af2, width, label='af2')
    
    # Set the x-axis labels
    ax.set_xticks(x)
    ax.set_xticklabels(dataset_types)
    
    # Set the y-axis label
    ax.set_ylabel(sel_col)
    if sel_col == 'cindex':
        ax.set_ylim([0.5, 1]) # 0.5 is the worst cindex value
    
    # Set the title and legend
    ax.set_title(f'Edge Type {sel_col} Difference')
    ax.legend()
    
    # Show the plot
    if show: plt.show()


if __name__ == '__main__':    
    csv = 'results/model_media/model_stats.csv'
    
    df = pd.read_csv(csv)
    df = pd.concat([df, pd.read_csv('results/model_media/old_model_stats.csv')]) # concat with old model results since we get the max value anyways...

    # create data, feat, and overlap columns for easier filtering.
    df['data'] = df['run'].str.extract(r'_(davis|kiba|PDBbind)', expand=False)
    df['feat'] = df['run'].str.extract(r'_(nomsa|msa|shannon)F_', expand=False)
    df['edge'] = df['run'].str.extract(r'_(binary|simple|anm|af2)E_', expand=False)
    df['ddp'] = df['run'].str.contains('DDP-')
    df['improved'] = df['run'].str.contains('IM_') # trail of model name will include I if "improved"
    df['batch_size'] = df['run'].str.extract(r'_(\d+)B_', expand=False)

    df.loc[df['run'].str.contains('EDIM') & df['run'].str.contains('nomsaF'), 'feat'] = 'ESM'
    df.loc[df['run'].str.contains('EDAIM'), 'feat'] += '-ESM'

    df['overlap'] = df['run'].str.contains('overlap')

    df[['run', 'data', 'feat', 'edge', 'batch_size', 'overlap']]

    fig1_pro_overlap(df)
    fig2_pro_feat(df)
    fig3_edge_feat(df)