import os
from typing import Tuple
from numbers import Number
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

def concordance_index(y_true, y_pred): #TODO: make this faster
    """
    Calculates the concordance index (CI) between two arrays of affinity values.
    """
    # sorting by y_true in ascending order and removing duplicates
    sorted_indices = np.argsort(y_true)
    y_true = y_true[sorted_indices]
    y_pred = y_pred[sorted_indices]
    
    # calculating concordance index
    sum = 0
    num_pairs = 0
    for i in range(len(y_true)):
        for j in range(i): # only need to loop through j < i
            if (y_true[i] > y_true[j]): # y[i] > y[j] is implied
                num_pairs += 1
                sum +=  1* (y_pred[i] > y_pred[j]) + 0.5 * (y_pred[i] == y_pred[j])
    return sum/num_pairs if num_pairs > 0 else 0
    
try:
    from lifelines.utils import concordance_index # faster implementation
except:
    pass

def get_metrics(log_y:np.array, log_z:np.array, save_results=True, 
                save_path='results/model_media',
                model_key='trained_davis_test',
                csv_file='results/model_media/DGraphDTA_stats.csv',
                show=True,
                title_prefix='') -> Tuple[Number]:
    """
    Display and save metrics

    Parameters
    ----------
    `log_y` : np.array
        The actual pkd values
    `log_z` : np.array
        Predicted pkd values
    `save_results` : bool, optional
        If Fase dont save anything, by default True
    `save_path` : str, optional
        Media save path for models, by default 'results/model_media'
    `model_key` : str, optional
        A discriptive name for the model (used for saving), by default 'trained_davis_test'
    `csv_file` : str, optional
        csv file for all model stats, by default 'results/model_media/DGraphDTA_stats.csv'
        
    `show` : bool, optional
        Whether or not to display out stats and plots, by default True
    `title_prefix` : str, optional
        Prefix for the title of the plots, by default ''.

    Returns
    -------
    Tuple[Number]
        Tuple for the stats (c_index, p_corr, s_corr, mse, mae, rmse)
    """
    
    plt.hist(log_y, bins=10, alpha=0.5)
    plt.hist(log_z, bins=10, alpha=0.5)
    plt.legend(['Experimental', model_key])
    plt.title(f'{title_prefix}Histogram of affinity values (pkd)')
    if save_results: plt.savefig(f'{save_path}/{model_key}_his.png')
    if show: plt.show()
    plt.clf()

    # scatter plot of affinity values
    # fitting a line
    m, b = np.polyfit(log_y, log_z, 1)
    plt.scatter(log_y, log_z, alpha=0.5)
    plt.plot(log_y, m*log_y + b, color='black', alpha=0.8)
    plt.xlabel('Experimental affinity value')
    plt.ylabel(f'{model_key} prediction')
    plt.title(f'{title_prefix}Scatter plot of affinity values (pkd)')

    if save_results: plt.savefig(f'{save_path}/{model_key}_scatter.png')
    if show: plt.show()
    plt.clf()


    # Stats
    c_index = concordance_index(log_y, log_z)
    p_corr = pearsonr(log_y, log_z)
    s_corr = spearmanr(log_y, log_z)

    # error
    mse = np.mean((log_y-log_z)**2)
    mae = np.mean(np.abs(log_y-log_z))
    rmse = np.sqrt(mse)
    
    if show:
        print(f"Concordance index: {c_index:.3f}")
        print(f"Pearson correlation: {p_corr[0]:.3f}")
        print(f"Pearson p-value: {p_corr[1]:.3f}")
        print(f"Spearman correlation: {s_corr[0]:.3f}")
        print(f"Spearman p-value: {s_corr[1]:.3f}")
        print(f"MSE: {mse:.3f}")
        print(f"MAE: {mae:.3f}")
        print(f"RMSE: {rmse:.3f}")

    # saving to csv file
    # creating stats csv if it doesnt exist
    if not os.path.exists(csv_file): 
        stats = pd.DataFrame(columns=['run', 'cindex', 'pearson', 'spearman', 'mse', 'mae', 'rmse'])
        stats.set_index('run', inplace=True)
        stats.to_csv(csv_file)

    # replacing existing record if run_num already exists
    if save_results:
        stats = pd.read_csv(csv_file, index_col=0)
        stats.loc[model_key] = [c_index, p_corr[0], s_corr[0], mse, mae, rmse]
        stats.to_csv(csv_file)
    
    return c_index, p_corr, s_corr, mse, mae, rmse



if __name__ == '__main__':
    #%% visualizing and analyzing docking results
    import matplotlib.pyplot as plt
    from scipy.stats import pearsonr, spearmanr
    import pandas as pd
    import numpy as np
    import random


    #%% cindex:
    if os.path.basename(os.getcwd()) == 'data_analysis':
        import os; os.chdir('../../') # for if running from src/data_analysis/
    print(os.getcwd())

    # CASF-2012 core dataset
    # core_2012_filter = []
    # with open('data/PDBbind/2012_core_data.lst', 'r') as f:
    #     for line in f.readlines():
    #         if '#' == line[0]: continue
    #         code = line[:4]
    #         core_2012_filter.append(code)
    # core_2012_filter = pd.DataFrame(core_2012_filter, columns=['PDBCode'])

    # filter = pd.read_csv('results/PDBbind/vina_out/run9.csv')['PDBCode'] #core_2012_filter['PDBCode']

    # Filter out for test split only
    np.random.seed(0)
    random.seed(0)
    df_x = pd.read_csv('data/PDBbind/kd_ki/X.csv') 
    pdbcodes = np.array(df_x['PDBCode'])
    random.shuffle(pdbcodes)
    _, filter = np.split(df_x['PDBCode'], [int(.8*len(df_x))])

    # %%
    save = False
    for run_num in [8,9]:
        print(f'run{run_num}:')
        y_path = 'data/PDBbind/kd_ki/Y.csv'
        vina_out = f'results/PDBbind/vina_out/run{run_num}.csv'
        save_path = 'results/PDBbind/media/kd_ki'

        ##%%
        vina_pred = pd.read_csv(vina_out)
        actual = pd.read_csv(y_path)
        vina_pred = vina_pred.merge(filter, on='PDBCode') # filter out for test split only
        
        mrgd = actual.merge(vina_pred, on='PDBCode')
        y = mrgd['affinity'].values
        z = mrgd['vina_kd(uM)'].values
        log_y = -np.log(y*1e-6)
        log_z = -np.log(z*1e-6)

        get_metrics(log_y, log_z, 
                    save_results=save,
                    save_path=save_path,
                    model_key=f'vina_{run_num}',
                    csv_file=f'{save_path}/vina_stats.csv',
                    
                    show=True,
                    title_prefix=f'run{run_num} - ',)
        
    
#%% plot bar graph of results by row (comparing vina to DGraphDTA)
# cols are: run,cindex,pearson,spearman,mse,mae,rmse
# df_res = pd.read_csv('results/model_media/DGraphDTA_stats.csv')[2:]
# df_res.sort_values(by='run', inplace=True)

# df_res.loc[-1] = ['vina', 0.68,0.508,0.520,17.812,3.427,4.220] # hard coded vina results

# for col in df_res.columns[1:]:
#     plt.figure()
#     bars = plt.bar(df_res['run'],df_res[col])
#     bars[0].set_color('green')
#     bars[2].set_color('green')
#     bars[-1].set_color('red')
#     plt.title(col)
#     plt.xlabel('run')
#     plt.xticks(rotation=30)
#     # plt.ylim((0.2, 0.8))
#     plt.show()