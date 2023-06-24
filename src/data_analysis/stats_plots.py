#%% visualizing and analyzing docking results
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
import pandas as pd
import numpy as np
import random

#%% cindex:
def concordance_index(y_true, y_pred):
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
    
##%% concordance index (This will take a while O(n^2))
try:
    from lifelines.utils import concordance_index
except:
    pass

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
    
    #NOTE: making sure to use the same data:
    vina_pred = vina_pred.merge(filter, on='PDBCode')

    ##%%
    mrgd = actual.merge(vina_pred, on='PDBCode')
    y = mrgd['affinity'].values
    z = mrgd['vina_kd(uM)'].values
    log_y = -np.log(y*1e-6)
    log_z = -np.log(z*1e-6)

    ##%% Statistics
    # calc concordance index 
    c_index = concordance_index(log_y, log_z)
    print(f"Concordance index: {c_index:.3f}")

    # pearson correlation
    p_corr = pearsonr(log_y, log_z)
    print(f"Pearson correlation: {p_corr[0]:.3f}")
    print(f"Pearson p-value: {p_corr[1]:.3f}")
    
    # spearman correlation
    s_corr = spearmanr(log_y, log_z)
    print(f"Spearman correlation: {s_corr[0]:.3f}")
    print(f"Spearman p-value: {s_corr[1]:.3f}")

    # error
    mse = np.mean((log_y-log_z)**2)
    mae = np.mean(np.abs(log_y-log_z))
    rmse = np.sqrt(mse)
    print(f"MSE: {mse:.3f}")
    print(f"MAE: {mae:.3f}")
    print(f"RMSE: {rmse:.3f}")

    ##%% saving results to csv file
    if save:
        # replacing existing record if run_num already exists
        stats = pd.read_csv(f'{save_path}/vina_stats.csv', index_col=0)

        stats.loc[run_num] = [c_index, p_corr[0], p_corr[1], mse, mae, rmse]
        stats = stats.sort_index()  # sorting by index

        # saving to csv
        stats.to_csv(f'{save_path}/vina_stats.csv')

    ##%% plotting histogram of affinity values
    plt.hist(log_y, bins=10, alpha=0.5)
    plt.hist(log_z, bins=10, alpha=0.5)
    plt.legend(['Experimental', 'Vina'])
    plt.title(f'run{run_num} - Histogram of affinity values (-log(Kd))')
    if save: plt.savefig(f'{save_path}/vina_{run_num}_hist.png')
    plt.show()

    # scatter plot of affinity values
    # fitting a line
    m, b = np.polyfit(log_y, log_z, 1)
    plt.scatter(log_y, log_z, alpha=0.5)
    plt.plot(log_y, m*log_y + b, color='black', alpha=0.8)
    plt.xlabel('Experimental affinity value')
    plt.ylabel('Vina prediction')
    plt.title(f'run{run_num} - Scatter plot of affinity values (-log(Kd))')

    if save: plt.savefig(f'{save_path}/vina_{run_num}_scatter.png')
    plt.show()
# %%
