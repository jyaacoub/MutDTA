#%% visualizing and analyzing docking results
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import pandas as pd
import numpy as np

run_num = 1

y_path = 'data/PDBbind/kd_ki/Y.csv'
vina_out = f'results/PDBbind/vina_out/run{run_num}.csv'

#%%
vina_pred = pd.read_csv(vina_out)
actual = pd.read_csv(y_path)

# %%
mrgd = actual.merge(vina_pred, on='PDBCode')
y = mrgd['affinity'].values
z = mrgd['vina_kd(uM)'].values
log_y = -np.log(y)
log_z = -np.log(z)

#%% calculating Concordance Index
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
    
#%% concordance index (This will take a while O(n^2))
try:
    from lifelines.utils import concordance_index
except:
    pass
c_index = concordance_index(log_y, log_z)
print(f"Concordance index: {c_index}")


# %% Statistics

# pearson correlation
p_corr = pearsonr(log_y, log_z)
print(f"Pearson correlation: {p_corr[0]}")
print(f"Pearson p-value: {p_corr[1]}")

# error
mse = np.mean((log_y-log_z)**2)
mae = np.mean(np.abs(log_y-log_z))
rmse = np.sqrt(mse)
print(f"MSE: {mse}")
print(f"MAE: {mae}")
print(f"RMSE: {rmse}")

# saving results to csv file
# replacing existing record if run_num already exists
stats = pd.read_csv(f'results/PDBbind/media/vina_stats.csv')
stats = stats[stats['run_num'] != run_num]

# adding to pd dataframe
stats.loc[len(stats)] = [run_num, c_index, p_corr[0], p_corr[1], mse, mae, rmse]
stats.run_num = stats.run_num.astype(int)

# sorting by run_num
stats = stats.sort_values(by='run_num')
# saving to csv
stats.to_csv(f'results/PDBbind/media/vina_stats.csv', index=False)

# %% plotting histogram of affinity values
plt.hist(log_y, bins=10, alpha=0.5)
plt.hist(log_z, bins=10, alpha=0.5)
plt.legend(['Experimental', 'Vina'])
plt.title('Histogram of affinity values (-log(Kd))')
plt.savefig(f'results/PDBbind/media/vina_{run_num}_hist.png')
plt.show()

# scatter plot of affinity values
# fitting a line
m, b = np.polyfit(log_y, log_z, 1)
plt.scatter(log_y, log_z, alpha=0.5)
plt.plot(log_y, m*log_y + b, color='black', alpha=0.8)
plt.xlabel('Experimental affinity value')
plt.ylabel('Vina prediction')
plt.title('Scatter plot of affinity values (-log(Kd))')

plt.savefig(f'results/PDBbind/media/vina_{run_num}_scatter.png')
plt.show()

# %%