#%%
import pandas as pd
import matplotlib.pyplot as plt

#%% TABLE FOR DATASET COUNTS 
def get_USED_dataset_counts(SPLITS_CSVS="/home/jyaacoub/projects/def-sushant/jyaacoub/MutDTA/splits/"):
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
def sequence_length_distribution(SPLITS_CSVS="/home/jyaacoub/projects/def-sushant/jyaacoub/MutDTA/splits", dataset_name='davis')
    """Distribution of sequences"""
    csvp=f"{SPLITS_CSVS}/{dataset_name}"
    df = pd.concat([
        pd.read_csv(f"{csvp}/test.csv", index_col=0),
        pd.read_csv(f"{csvp}/train0.csv", index_col=0),
        pd.read_csv(f"{csvp}/val0.csv", index_col=0)
    ])
    df['len'] = df.prot_seq.str.len()
    
    n, bins, patches = plt.hist(df['len'], bins=20)
    plt.figure(figsize(15))
    # Set labels and title
    plt.xlabel('Protein Sequence length')
    plt.ylabel('Frequency')
    plt.title(f'Histogram of Protein Sequence length ({dataset_name})')

sequence_length_distribution()
