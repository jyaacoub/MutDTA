import os
from typing import Tuple
from numbers import Number
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from lifelines.utils import concordance_index


def get_metrics(log_y, log_z, save_results=True, 
                save_path='results/model_media',
                model_key='trained_davis_test',
                csv_file='results/model_media/DGraphDTA_stats.csv',
                show=True) -> Tuple[Number]:
    """
    Display and save metrics

    Parameters
    ----------
    log_y : np.array
        The actual pkds
    log_z : np.array
        The prediced pkds
    save_results : bool, optional
        If False dont save anything, only display. Defaults to True.
    save_path : str, optional
        Media save path for models. Defaults to 'results/model_media'.
    model_key : str, optional
        A discriptive name for the model. Defaults to 'trained_davis_test'.
    csv_file : str, optional
        csv file for all the model stats. Defaults to 'results/model_media/DGraphDTA_stats.csv'.
    """
    
    plt.hist(log_y, bins=10, alpha=0.5)
    plt.hist(log_z, bins=10, alpha=0.5)
    plt.legend(['Experimental', model_key])
    plt.title(f'Histogram of affinity values (pkd)')
    if save_results: plt.savefig(f'{save_path}/{model_key}_his.png')
    if show: plt.show()

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

    # Stats
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



