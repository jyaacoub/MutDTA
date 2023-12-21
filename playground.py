# %%
from src.utils.mmseq2 import MMseq2Runner
from src.data_analysis.stratify_protein import map_davis_to_kinbase
import pandas as pd 
from glob import glob
import os


## CLUSTERING PARAMETERS
CLUST_OPTION = '4sens_9c_5cov'

## PLOTTING PARAMETERS
GET_MEAN = True

## DATASET PARAMETERS
for DATASET in  ['pdbbind', 'kiba', 'davis']:

    ## Processing for getting correct paths based on dataset:
    if DATASET == 'davis' or DATASET == 'kiba':
        XY_csv_path = f'../data/DavisKibaDataset/{DATASET}/'
    elif DATASET == 'pdbbind':
        XY_csv_path = f'../data/PDBbindDataset/'
    else:
        raise ValueError('Invalid dataset')
        
    XY_csv_path += '/nomsa_binary_original_binary/full/XY.csv'
    clust_dir = f'../data/misc/{DATASET}_clustering'

    if DATASET == 'davis':
        # tsvp = f'{clust_dir}//tsvs/davisDB_4sens_198clust.tsv'
        # tsvp = f'{clust_dir}/tsvs/davisDB_4sens_5cov_9c.tsv' # 49 clusters
        tsvp = f'{clust_dir}/tsvs/{DATASET}DB_4sens_9c_5cov.tsv' # 198 clusters
    elif DATASET == 'kiba':
        tsvp = f'{clust_dir}/tsvs/{DATASET}DB_4sens_9c_5cov.tsv'
    elif DATASET == 'pdbbind':
        tsvp = f'{clust_dir}/tsvs/{DATASET}DB_4sens_9c_5cov.tsv'
    else:
        raise ValueError('Invalid dataset')

    assert os.path.exists(tsvp), f"TSV file {os.path.basename(tsvp)} does not exist. Please run clustering first."

    # read tsv
    df = pd.read_csv(tsvp, sep='\t', header=None)
    # rename cols
    df.columns = ['rep', 'member']

    clusters = df.groupby('rep')['member'].apply(list).to_dict()
    print(len(clusters), 'clusters')

    # group clusters with less than 5 members into one cluster (cluster 0)
    clusters_new = {}
    for k in clusters.keys():
        if len(clusters[k]) == 0: # remove outlier clusters
            continue
            #clusters_new[0] = clusters_new.get(0, []) + clusters[k]
        else:
            clusters_new[k] = clusters[k]
            

    clusters = clusters_new

    # Rename clusters keys to be ints 
    clusters = {i: set(clusters[k]) for i, k in enumerate(clusters)}


    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np

    sns.set_style('darkgrid')

    # get count of every subgroup
    # cols are code,SMILE,prot_seq,pkd,prot_id
    df_full = pd.read_csv(XY_csv_path,
                        index_col='code')

    prot_counts = df_full.index.value_counts()


    cluster_counts = {} # {cluster: count, ...}
    for idx in df_full.index:
        for k in clusters.keys():
            if idx in clusters[k]:
                cluster_counts[k] = cluster_counts.get(k, 0) + 1
                break
        
    # get subgroup mse
    subset = 'test' #'test'
    # plot mse vs mse for EDI and DG
    model_cluster_mse = {} # {model_type: {cluster: mse, ...}, ...}
    for model_type in ['EDI', 'DG']:
        d = DATASET if DATASET != 'pdbbind' else 'PDBbind'
        
        paths = glob(f'results/model_media/{subset}_set_pred/{model_type}M_{d}*')
        assert len(paths) == 5, f"Incorrect number of folds for {model_type} model. {paths}"

        paths = sorted(paths) # should be in order of fold 0, 1, 2, 3, 4
        
        data_clust = {} # {cluster: [mse1, mse2, ...], ...}
        for fold in range(5):
            pred = pd.read_csv(paths[fold], index_col='name')
            
            for k in clusters.keys():
                # get mse for cluster
                matched = pred[pred.index.isin(clusters[k])]
                if matched.empty:
                    continue
                mse = ((matched.pred - matched.actual)**2).mean()
            
                # add main mse to dict
                data_clust[k] = data_clust.get(k, []) + [mse]

        # merge counts with mse
        model_cluster_mse[model_type] = data_clust.copy()
        

    # plot mse vs mse for EDI and DG
    # verify that clusters are present in both models
    # if not, remove from both
    for k in list(model_cluster_mse['EDI'].keys()):
        assert k in model_cluster_mse['DG'], "cluster not in both models, this is likely due to mismatched subsets being used."


    x,y,z = [], [], []
    for k in model_cluster_mse['EDI'].keys():
        assert len(model_cluster_mse['EDI'][k]) == len(model_cluster_mse['DG'][k]), "Cluster size mismatch. Are you using the same number of folds for both models?"
        if GET_MEAN:
            x.append(np.mean(model_cluster_mse['EDI'][k]))
            y.append(np.mean(model_cluster_mse['DG'][k]))
            z.append(k)
        else:
            x += model_cluster_mse['EDI'][k]
            y += model_cluster_mse['DG'][k]
            z += [k]*len(model_cluster_mse['EDI'][k])

    sns.set_style('darkgrid')

    plt.figure(figsize=(10,5))
    sns.scatterplot(x=x, y=y, hue=z, palette='tab20')

    # plot line at y=x for reference
    plt.plot([0,max(x)], [0,max(x)], color='black', linestyle='--')

    plt.xlabel('EDI MSE')
    plt.ylabel('DG MSE')
    plt.title(f'MSE of clusters for EDI and DG models ({DATASET} {subset} set)')

    # remove legend
    plt.legend([],[], frameon=False)

    plt.show()
    plt.clf()    

    # %%
