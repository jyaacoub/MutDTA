from typing import Tuple
from collections import OrderedDict
from scipy.stats import ttest_ind
import pandas as pd
import numpy as np
from src import config as cfg


def count_missing_res(pdb_file: str) -> Tuple[int,int]:
    """
    Returns the number of missing residues

    Parameters
    ----------
    `pdb_file` : str
        The path to the PDB file

    Returns
    -------
    Tuple[int,int]
        Gap instances, number of missing residues
    """

    chains = OrderedDict() # chain dict of dicts
    # read and filter
    with open(pdb_file, 'r') as f:
        lines = f.readlines()

        ter = 0 # chain terminator
        chains[0] = [] # res counts for first chain
        for line in lines:
            if (line[:6].strip() == 'TER'): # TER indicates new chain "terminator"
                ter += 1
                chains[ter] = [] # res count
            
            if (line[:6].strip() != 'ATOM'): continue # skip non-atom lines

            # appending res count to list
            curr_res = int(line[22:26])
            chains[ter].append(curr_res)
    
    # counting number of missing residues
    num_missing = 0
    num_gaps = 0
    for c in chains.values():
        curr, prev = None, None
        for res_num in sorted(c):
            prev = curr
            curr = res_num
            if (prev is not None) and \
                (curr != prev and curr != prev+1):
                num_missing += curr - prev
                num_gaps +=1
    
    return num_gaps, num_missing

###################################
#### FOR MUTATION ANALYSIS ########
def get_mut_count(df):
    """creates column called n_mut for the number of mutations a protein has depending on its id"""
    df['pdb'] = df['prot_id'].str.split('_').str[0]
    n_mut = []
    for code, prot_id in df[['prot_id']].iterrows():
        if '_wt' in code:
            n_mut.append(0)
        else:
            n_mut.append(len(prot_id[0].split('-')))
            
    df['n_mut'] = n_mut
    return df

def generate_markdown(results, names=None, verbose=False, thresh_sig=False, cindex=False):
    """
    generates a markdown given a list or single df containing metrics from get_metrics
    
    example usage:
    ```
        _, p_corr, s_corr, mse, mae, rmse = get_metrics(true_dpkd1, pred_dpkd1)
        result = [[p_corr[0], s_corr[0], mse, mae, rmse]]
        generate_markdown(result)
    ```
    """
    n_groups = len(results)
    names = names if len(names)>0 else [str(i) for i in range(n_groups)]
    # Convert results to DataFrame
    results_df = [None for _ in range(n_groups)]
    md_table = None
    cols = ['cindex'] if cindex else []
    cols += ['pcorr', 'scorr', 'mse', 'mae', 'rmse']
    for i, r in enumerate(results):
        df = pd.DataFrame(r, columns=cols)

        mean = df.mean(numeric_only=True)
        std = df.std(numeric_only=True)
        results_df[i] = df
        
        # calculate standard error:
        se = std / np.sqrt(len(df))
        
        # formating for markdown table:
        combined = mean.map(lambda x: f"{x:.3f}") + " $\pm$ " + se.map(lambda x: f"{x:.3f}")
        md_table = combined if md_table is None else pd.concat([md_table, combined], axis=1)

    if n_groups == 2: # no support for sig  if groups are more than 2
        # two-sided t-tests for significance
        ttests = {col: ttest_ind(results_df[0][col], results_df[1][col]) for col in results_df[0].columns}
        if thresh_sig:
            sig = pd.Series({col: '*' if ttests[col].pvalue < 0.05 else '' for col in results_df[0].columns})
        else:
            sig =pd.Series({col: f"{ttests[col].pvalue:.4f}" for col in results_df[0].columns})

        md_table = pd.concat([md_table, sig], axis=1)
        md_table.columns = [*names, 'p-val']
    else:
        md_table = pd.DataFrame(md_table)
        md_table.columns = names

    md_output = md_table.to_markdown()
    if verbose: print(md_output)
    return md_table

def combine_dataset_pids(data_dir=cfg.DATA_ROOT, 
                         dbs=[cfg.DATA_OPT.davis, cfg.DATA_OPT.kiba, cfg.DATA_OPT.PDBbind],
                         target='nomsa_aflow_gvp_binary', subset='full', 
                         xy='cleaned_XY.csv'):
    df_all = None
    dir_p = {'davis':'DavisKibaDataset/davis', 
           'kiba': 'DavisKibaDataset/kiba',
           'PDBbind': 'PDBbindDataset'}
    dbs = {d.value: f'{data_dir}/{dir_p[d]}/{target}/{subset}/{xy}' for d in dbs}
    
    for DB, fp in dbs.items():
        print(DB, fp)
        df = pd.read_csv(fp, index_col=0)
        df['pdb_id'] = df.prot_id.str.split("_").str[0]
        df = df[['prot_id', 'prot_seq']].drop_duplicates(subset='prot_id')
        
        df['seq_len'] = df['prot_seq'].str.len()
        df['db'] = DB
        df.reset_index(inplace=True)
        
        df = df[['db','code', 'prot_id', 'seq_len', 'prot_seq']] # reorder them.
        df.index.name = 'db_idx'
        df_all = df if df_all is None else pd.concat([df_all, df], axis=0)
    
    return df_all

if __name__ == '__main__':
    #NOTE: the following is code for stratifying AutoDock Vina results by 
    # missing residues to identify if there is a correlation between missing 
    # residues and performance. 
    #%% get metrics across different quality pdbs
    import pandas as pd
    import numpy as np


    import matplotlib.pyplot as plt
    from tqdm import tqdm

    from src.analysis import get_save_metrics, count_missing_res
    #%% 
    pdb_path = lambda x: f'/home/jyaacoub/projects/data/refined-set/{x}/{x}_protein.pdb'

    run_num=9
    y_path = 'data/PDBbind/kd_ki/Y.csv'
    vina_out = f'results/PDBbind/vina_out/run{run_num}.csv'
    save_path = 'results/PDBbind/media/kd_ki'

    ##%%
    vina_pred = pd.read_csv(vina_out) # vina_deltaG(kcal/mol),vina_kd(uM)
    vina_pred['vina_pkd'] = -np.log(vina_pred['vina_kd(uM)']*1e-6)
    vina_pred.drop(['vina_deltaG(kcal/mol)', 'vina_kd(uM)'], axis=1, inplace=True)

    actual = pd.read_csv(y_path)      # affinity (in uM)
    actual['actual_pkd'] = -np.log(actual['affinity']*1e-6)
    actual.drop('affinity', axis=1, inplace=True)

    mrgd = actual.merge(vina_pred, on='PDBCode')

    #%% 
    col = 'missing_res'
    # col='res'
    # res_file = '/home/jyaacoub/projects/data/refined-set/index/INDEX_refined_set.2020' 
    # with open(res_file, 'r') as f:
    #     lines = f.readlines()
    #     print(len(lines))
    #     res_dict = {}
    #     for l in lines:
    #         if l[0] == '#': continue
    #         code = l[0:5].strip()
    #         res = float(l[5:10])
    #         res_dict[code] = res

    # df_ = pd.DataFrame.from_dict(res_dict, orient='index', columns=['res'])
    missing = [count_missing_res(pdb_path(code))[1] for code in tqdm(mrgd['PDBCode'], 'Getting missing count')]

    mrgd[col] = missing

    #%%
    df_ = mrgd.sort_values(by=col)

    num_bins=10
    bin_size = int(len(df_)/num_bins)
    print(f'bin size: {bin_size}')
    bins = {} # dic of pd dfs and avg for col
    for i in range(num_bins):
        df_bin = df_.iloc[i*bin_size:(i+1)*bin_size]
        avg = df_bin[col].max()
        bins[i] = (avg, df_bin)

    # %% 
    metrics = []
    for i in range(num_bins):
        df_b = bins[i][1]
        pkd_y, pkd_z = df_b['actual_pkd'].to_numpy(), df_b['vina_pkd'].to_numpy()
        print(f'\nBin {i}, size: {len(df_b)}, {col}: {bins[i][0]}')
        metrics.append(get_save_metrics(pkd_y, pkd_z, save_figs=False, show=False))
    print("sample metrics:", *metrics[0])

    # %%
    options = ['c_index', 'p_corr', 's_corr', 'mse', 'mae', 'rmse']
    choice = 0
    # cindex scatter by col
    bin_ = [str(v[0]) for v in bins.values()]
    cindex = [m[0] for m in metrics]
    plt.bar(bin_, cindex)
    plt.ylim((0.5,0.9))
    plt.xlabel('Avg missing res')
    if num_bins > 20: plt.xticks(rotation=30)
    plt.ylabel('C-index')
    plt.title(f'Vina cindex vs {col}')

    # %%