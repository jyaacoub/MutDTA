"""
This file contains functions for evaluating docking and affinity prediction model results.
"""
# %%
import os
from typing import Tuple
from numbers import Number
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr


def pdb_wRMSD(pdb_file: str, pred_file: str) -> Number:
    """
    "An important extension of the RMSD measure, the weighted RMSD (wRMSD),
    allows focusing on selected atomic subsets, for example, downplaying the
    regions known to be inherently unstructured."
        
        - https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4321859/
    """
    raise NotImplementedError


def pdb_RMSD(pdb_file: str, pred_file: str) -> Number:
    """
    Returns the root mean square deviation (RMSD) between two structures.
    
    From (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4321859/) the RMSD of the 
    lowest-energy structures wrt the experiments are used to measure predictive 
    performance for docking.
    
    The average RMSD of the predicted ligand conformations across all complexes 
    is used as the final performance metric for the docking task.
    
    Parameters
    ----------
    `pdb_file` : str
        The path to the actual experimental structure.
    `pred_file` : str
        Path to the predicted structure by docking.

    Returns
    -------
    Number
        The RMSD between the two structures.
    """
    raise NotImplementedError 
# %%
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

def get_metrics(y_true: np.array, y_pred: np.array, save_results=True, 
                save_path='results/model_media',
                model_key='trained_davis_test',
                csv_file='results/model_media/DGraphDTA_stats.csv',
                show=True,
                title_prefix='') -> Tuple[Number]:
    """
    Display and save metrics

    Parameters
    ----------
    `y_true` : np.array
        The actual pkd values
    `y_pred` : np.array
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
    plt.clf()
    plt.hist(y_true, bins=10, alpha=0.5)
    plt.hist(y_pred, bins=10, alpha=0.5)
    plt.legend(['Experimental', model_key])
    plt.title(f'{title_prefix}Histogram of affinity values (pkd)')
    if save_results: plt.savefig(f'{save_path}/{model_key}_his.png')
    if show: plt.show()
    plt.clf()

    # scatter plot of affinity values
    # fitting a line
    m, b = np.polyfit(y_true, y_pred, 1)
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot(y_true, m*y_true + b, color='black', alpha=0.8)
    plt.xlabel('Experimental affinity value')
    plt.ylabel(f'{model_key} prediction')
    plt.title(f'{title_prefix}Scatter plot of affinity values (pkd)')

    if save_results: plt.savefig(f'{save_path}/{model_key}_scatter.png')
    if show: plt.show()
    plt.clf()


    # Stats
    c_index = concordance_index(y_true, y_pred)
    p_corr = pearsonr(y_true, y_pred)
    s_corr = spearmanr(y_true, y_pred)

    # error
    mse = np.mean((y_true-y_pred)**2)
    mae = np.mean(np.abs(y_true-y_pred))
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

