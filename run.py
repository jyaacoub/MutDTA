#%% testing download of sequences from FASTA files
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import pandas as pd
import numpy as np

y_path = 'data/PDBbind/kd_ki/Y.csv'
vina_out = 'results/PDBbind/vina_out/run1.csv'

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
    #TODO:make faster
    loops through all possible pairs i,j where Y[i] > Y[j]
    """
    # sort y_true and y_pred by y_true
    Y = y_true
    P = y_pred
    sum = 0
    num_pairs = 0
    
    # loop through all combinations of pairs
    for i in range(len(Y)):
        for j in range(len(Y)):
            if(Y[i] > Y[j]):
                num_pairs += 1
                sum +=  1* (P[i] > P[j]) + 0.5 * (P[i] == P[j]) # step function 1 if x > 0, 0.5 if x == 0, 0 if x < 0
            
    if num_pairs != 0:
        print(f"\tnum_pairs: {num_pairs}")
        return sum/num_pairs
    else:
        return 0
    
def get_cindex(Y, P): 
    # This came from DeepDTA and is wrong (https://github.com/hkmztrk/DeepDTA/blob/2c9cbafdfb383f2f03bcea4b231b90a072e65b15/source/emetrics.py#L25)
    # https://github.com/hkmztrk/DeepDTA/issues/18
    summ = 0
    pair = 0
    
    for i in range(len(Y)):
        for j in range(i):
            if(Y[i] > Y[j]):
                pair += 1
                summ +=  1* (P[i] > P[j]) + 0.5 * (P[i] == P[j])
        
            
    if pair != 0:
        print(f"\tnum_pairs: {pair}")
        return summ/pair
    else:
        return 0
    
# concordance index (This will take a while O(n^2))
c_index = concordance_index(log_y, log_z)
print(f"Concordance index: {c_index}")
print(f'c index: {get_cindex(log_y, log_z)}')

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

# %% plotting histogram of affinity values
plt.hist(log_y, bins=10, alpha=0.5)
plt.hist(log_z, bins=10, alpha=0.5)
plt.legend(['Experimental', 'Vina'])
plt.title('Histogram of affinity values (-log(Kd))')
plt.show()

# scatter plot of affinity values
# fitting a line
m, b = np.polyfit(log_y, log_z, 1)
plt.scatter(log_y, log_z, alpha=0.5)
plt.plot(log_y, m*log_y + b, color='black', alpha=0.8)
plt.xlabel('Experimental affinity value')
plt.ylabel('Vina prediction')
plt.title('Scatter plot of affinity values (-log(Kd))')
plt.show()

# %%