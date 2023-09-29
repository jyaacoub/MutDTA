import matplotlib.pyplot as plt
import pandas as pd
import numpy as np



# Figure 1 - Protein overlap cindex difference (nomsa)
def fig1_pro_overlap(df, sel_col='cindex', verbose=False):
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
        t_overlap_vals = group[group['overlap']][sel_col].values
        if len(t_overlap_vals) == 0:
            t_overlap.append(0)
        elif len(t_overlap_vals) == 1:
            t_overlap.append(t_overlap_vals[0])
        else:
            raise IndexError('Too many overlap values, filter is too broad.')
        
        # no overlap
        f_overlap_vals = group[~group['overlap']][sel_col].values
        if verbose: print(f_overlap_vals)
        if len(f_overlap_vals) == 0:
            f_overlap.append(0)
        elif len(f_overlap_vals) == 1:
            f_overlap.append(f_overlap_vals[0])
        else:
            raise IndexError('Too many overlap values, filter is too broad.')
        dataset_types.append(dataset)

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
    ax.set_ylabel('cindex')
    ax.set_ylim([0.5, 1]) # 0.5 is the worst cindex value

    # Set the title and legend
    ax.set_title('Protein Overlap cindex Difference (nomsa)')
    ax.legend()

    # Show the plot
    plt.show()

# Figure 2 - Feature type cindex difference
# Features -> nomsa, msa, shannon, and esm
def fig2_pro_feat(df, verbose=False):
    # comparing nomsa, msa, shannon, and esm
    # group by data type
    
    # this will capture multiple models per dataset (different LR, batch size, etc)
    #   Taking the max cindex value for each dataset will give us the best model for each dataset
    grouped_df = df[(df['edge'] == 'binary')
                    & (~df['overlap'])].groupby(['data'])
    
    # each group is a dataset with 4 bars (nomsa, msa, shannon, esm)
    for group_name, group_data in grouped_df:
        if verbose: print(f"\nGroup Name: {group_name}")
        if verbose: print(group_data[['cindex', 'mse', 'feat']])

    # these groups are spaced by the data type, physically grouping bars of the same dataset together.
    # Initialize lists to store cindex values for each dataset type
    nomsa = []
    msa = []
    shannon = []
    esm = []
    dataset_types = []
    
    for dataset, group in grouped_df:
        if verbose: print('')
        if verbose: print(group[['cindex', 'mse', 'overlap', 'data']])
        
        nomsa_v = group[group['feat'] == 'nomsa']['cindex'].max() # NOTE: take min for mse
        msa_v = group[group['feat'] == 'msa']['cindex'].max()
        shannon_v = group[group['feat'] == 'shannon']['cindex'].max()
        ESM_v = group[group['feat'] == 'ESM']['cindex'].max()
        
        # appending if not nan else 0
        nomsa.append(nomsa_v if not np.isnan(nomsa_v) else 0)
        msa.append(msa_v if not np.isnan(msa_v) else 0)
        shannon.append(shannon_v if not np.isnan(shannon_v) else 0)
        esm.append(ESM_v if not np.isnan(ESM_v) else 0)
        dataset_types.append(dataset)
        
    # Create an array of x positions for the bars
    x = np.arange(len(dataset_types))
    
    # Set the width of the bars
    width = 0.2
    
    # Create a bar plot with 4 bars for each dataset type
    fig, ax = plt.subplots()
    bar1 = ax.bar(x - width, nomsa, width, label='nomsa')
    bar2 = ax.bar(x, msa, width, label='msa')
    bar3 = ax.bar(x + width, shannon, width, label='shannon')
    bar4 = ax.bar(x + width*2, esm, width, label='esm')
    
    # Set the x-axis labels
    ax.set_xticks(x)
    ax.set_xticklabels(dataset_types)
    
    # Set the y-axis label
    ax.set_ylabel('cindex')
    ax.set_ylim([0.5, 1])
    
    # Set the title and legend
    ax.set_title('Feature Type cindex Difference')
    ax.legend()
    
    # Show the plot
    plt.show()

# Figure 3 - Edge type cindex difference
# Edges -> binary, simple, anm, af2
def fig3_edge_feat(df, verbose=False):
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
        
        binary_v = group[group['edge'] == 'binary']['cindex'].max() # NOTE: take min for mse
        simple_v = group[group['edge'] == 'simple']['cindex'].max()
        anm_v = group[group['edge'] == 'anm']['cindex'].max()
        af2_v = group[group['edge'] == 'af2']['cindex'].max()
        
        # appending if not nan else 0
        binary.append( binary_v if not np.isnan(binary_v) else 0)
        simple.append( simple_v if not np.isnan(simple_v) else 0)
        anm.append(anm_v if not np.isnan(anm_v) else 0)
        af2.append(af2_v if not np.isnan(af2_v) else 0)
        dataset_types.append(dataset)
        
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
    ax.set_ylabel('cindex')
    ax.set_ylim([0.5, 1])
    
    # Set the title and legend
    ax.set_title('Edge Type cindex Difference')
    ax.legend()
    
    # Show the plot
    plt.show()


if __name__ == '__main__':    
    csv = 'results/model_media/model_stats.csv'

    df = pd.read_csv(csv)

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