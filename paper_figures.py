#%%
import pandas as pd
import matplotlib.pyplot as plt

#%% TABLE FOR DATASET COUNTS 
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

#%% SEQUENCE LENGTH DISTRIBUTION
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


###########################################
#%% MODEL RESULTS
###########################################
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