import os
from typing import Tuple
from numbers import Number
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from lifelines.utils import concordance_index


def get_metrics(log_y:np.array, log_z:np.array, save_results=True, 
                save_path='results/model_media',
                model_key='trained_davis_test',
                csv_file='results/model_media/DGraphDTA_stats.csv',
                show=True) -> Tuple[Number]:
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

    Returns
    -------
    Tuple[Number]
        Tuple for the stats (c_index, p_corr, s_corr, mse, mae, rmse)
    """
    
    plt.hist(log_y, bins=10, alpha=0.5)
    plt.hist(log_z, bins=10, alpha=0.5)
    plt.legend(['Experimental', model_key])
    plt.title(f'Histogram of affinity values (pkd)')
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
    plt.title(f'Scatter plot of affinity values (pkd)')

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
    if not os.path.exists(f'{save_path}/DGraphDTA_stats.csv'): 
        stats = pd.DataFrame(columns=['run', 'cindex', 'pearson', 'spearman', 'mse', 'mae', 'rmse'])
        stats.set_index('run', inplace=True)
        stats.to_csv(csv_file)

    # replacing existing record if run_num already exists
    if save_results:
        stats = pd.read_csv(csv_file, index_col=0)
        stats.loc[model_key] = [c_index, p_corr[0], s_corr[0], mse, mae, rmse]
        stats.to_csv(csv_file)
    
    return c_index, p_corr, s_corr, mse, mae, rmse



