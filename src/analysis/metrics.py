"""
This file contains functions for evaluating docking and affinity prediction model results.
"""
# %%
import os
from typing import Tuple
from numbers import Number
from matplotlib.ticker import MaxNLocator
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
def concordance_index(y_true, y_pred) -> float:
    """Faster implementation of cindex by utilizing vectorization and numpy"""
    sorted_indices = np.argsort(y_true)
    y_true_sorted = y_true[sorted_indices]
    y_pred_sorted = y_pred[sorted_indices]
    
    # Calculate concordant and discordant pairs
    # where i > j is all we care about
    i, j = np.where(y_true_sorted[:,np.newaxis] > y_true_sorted)
    pairs_gt = np.sum(y_pred_sorted[i] > y_pred_sorted[j])
    pairs_eq = np.sum(y_pred_sorted[i] == y_pred_sorted[j])*0.5
    
    # Calculate the concordance index
    c_index = (pairs_gt + pairs_eq)/len(i)

    return c_index
    
    
try:
    from lifelines.utils import concordance_index # faster implementation
except:
    pass

def get_metrics(y_true, y_pred, verbose=False):
    # get metrics:
    try:
        # cindex throws zero division error if all values are the same
        c_index = concordance_index(y_true, y_pred)
    except ZeroDivisionError:
        c_index = -1
        print('ZeroDivisionError: cindex set to -1')
        
    p_corr = pearsonr(y_true, y_pred)
    s_corr = spearmanr(y_true, y_pred)

    # error
    mse = np.mean((y_true-y_pred)**2)
    mae = np.mean(np.abs(y_true-y_pred))
    rmse = np.sqrt(mse)
    
    if verbose:
        print(f"Concordance index: {c_index:.3f}")
        print(f"Pearson correlation: {p_corr[0]:.3f}")
        print(f"Pearson p-value: {p_corr[1]:.3f}")
        print(f"Spearman correlation: {s_corr[0]:.3f}")
        print(f"Spearman p-value: {s_corr[1]:.3f}")
        print(f"MSE: {mse:.3f}")
        print(f"MAE: {mae:.3f}")
        print(f"RMSE: {rmse:.3f}")
   

    return c_index, p_corr, s_corr, mse, mae, rmse

def get_save_metrics(y_true: np.array, y_pred: np.array, save_figs=True, save_data=True,
                save_path='results/model_media',
                model_key='Predictions',
                csv_file='results/model_media/DGraphDTA_stats.csv',
                show=True,
                title_prefix='', title_postfix='affinity values (pkd)',
                dataset='test', # for discriminating between test and val
                logs=None) -> Tuple[Number]:
    """
    Display and save metrics

    Parameters
    ----------
    `y_true` : np.array
        The actual pkd values
    `y_pred` : np.array
        Predicted pkd values
    `save_figs` : bool, optional
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
    dataset = '' if dataset == 'test' else dataset # To maintain compatibility with prev figs
    
    ############ plot histogram of data distribution of pred and true #################
    plt.clf()
    plt.hist(y_true, bins=10, alpha=0.5)
    plt.hist(y_pred, bins=10, alpha=0.5)
    plt.legend(['Experimental', model_key])
    plt.title(f'{title_prefix}Histogram of {title_postfix}')
    if save_figs: plt.savefig(f'{save_path}/{model_key}_his{dataset}.png')
    if show: plt.show()
    plt.clf()

    ################ scatter plot of affinity values x=true y=pred ###############
    # fitting a line
    m, b = np.polyfit(y_true, y_pred, 1)
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot(y_true, m*y_true + b, color='black', alpha=0.8)
    plt.xlabel('Experimental affinity value')
    plt.ylabel(f'{model_key} prediction')
    plt.grid(True)
    plt.title(f'{title_prefix}Scatter plot of {title_postfix}')

    if save_figs: plt.savefig(f'{save_path}/{model_key}_scatter{dataset}.png')
    if show: plt.show()
    plt.clf()

    ############ display train val plot ###########
    if logs is not None: 
        ax = plt.figure().gca()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        index = np.arange(1, len(logs['train_loss'])+1)
        plt.plot(index, logs['train_loss'], label='train')
        plt.plot(index, logs['val_loss'], label='val')
        plt.legend()
        plt.title(f'{model_key} Loss')
        plt.xlabel('Epoch')
        # plt.xticks(range(0,NUM_EPOCHS+1, 2))
        # plt.xlim(0, NUM_EPOCHS)
        plt.ylabel('Loss')
        if save_figs: plt.savefig(f'{save_path}/{model_key}_loss{dataset}.png')
        if show: plt.show()
    plt.clf()
    
    ########### saving metrics to csv file ###########
    # get metrics:
    c_index, p_corr, s_corr, mse, mae, rmse = get_metrics(y_true, y_pred, show)
    
    if save_data:
        # creating stats csv if it doesnt exist or empty/incomplete header
        if not os.path.exists(csv_file) or os.path.getsize(csv_file) < 40:
            stats = pd.DataFrame(columns=['run', 'cindex', 'pearson', 'spearman', 'mse', 'mae', 'rmse'])
            stats.set_index('run', inplace=True)
            stats.to_csv(csv_file)

        # replacing existing record if run_num already exists
        stats = pd.read_csv(csv_file, index_col=0)
        stats.loc[model_key] = [c_index, p_corr[0], s_corr[0], mse, mae, rmse]
        stats.to_csv(csv_file)    
    
    return c_index, p_corr, s_corr, mse, mae, rmse
