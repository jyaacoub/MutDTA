# %%
########################################################################
########################## VIOLIN PLOTTING #############################
########################################################################
import logging
from matplotlib import pyplot as plt

from src.analysis.figures import prepare_df, fig_combined, custom_fig

dft = prepare_df('./results/v115/model_media/model_stats.csv')
dfv = prepare_df('./results/v115/model_media/model_stats_val.csv')

models = {
    'DG': ('nomsa', 'binary', 'original', 'binary'),
    'esm': ('ESM', 'binary', 'original', 'binary'), # esm model
    'aflow': ('nomsa', 'aflow', 'original', 'binary'),
    # 'gvpP': ('gvp', 'binary', 'original', 'binary'),
    'gvpL': ('nomsa', 'binary', 'gvp', 'binary'),
    # 'aflow_ring3': ('nomsa', 'aflow_ring3', 'original', 'binary'),
    'gvpL_aflow': ('nomsa', 'aflow', 'gvp', 'binary'),
    # 'gvpL_aflow_rng3': ('nomsa', 'aflow_ring3', 'gvp', 'binary'),
    #GVPL_ESMM_davis3D_nomsaF_aflowE_48B_0.00010636872718329864LR_0.23282479481785903D_2000E_gvpLF_binaryLE
    # 'gvpl_esm_aflow': ('ESM', 'aflow', 'gvp', 'binary'),
}

fig, axes = fig_combined(dft, datasets=['davis'], fig_callable=custom_fig,
             models=models, metrics=['cindex', 'mse'],
             fig_scale=(10,5), add_stats=True, title_postfix=" test set performance", box=True, fold_labels=True)
plt.xticks(rotation=45)

fig, axes = fig_combined(dfv, datasets=['davis'], fig_callable=custom_fig,
             models=models, metrics=['cindex', 'mse'],
             fig_scale=(10,5), add_stats=True, title_postfix=" validation set performance", box=True, fold_labels=True)
plt.xticks(rotation=45)