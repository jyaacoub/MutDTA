# %%
import matplotlib.pyplot as plt
import seaborn as sns 

from src.utils import config as cfg
from src.data_analysis.figures import (prepare_df, fig1_pro_overlap, 
                                    fig2_pro_feat, fig3_edge_feat, 
                                    fig4_pro_feat_violin, fig5_edge_feat_violin, fig_combined)


# %%
df = prepare_df(csv_p=cfg.MODEL_STATS_CSV, old_csv_p="results/model_media/old_model_stats.csv")

#%% display figures
verbose = False
sns.set(style="darkgrid")
fig_combined(df, fig_callable=fig4_pro_feat_violin, verbose=verbose)
plt.savefig("results/figures/fig_combined_proViolin_CI-MSE.png", dpi=300)

#%%
fig_combined(df, fig_callable=fig5_edge_feat_violin, verbose=verbose)
plt.savefig("results/figures/fig_combined_edgeViolin_CI-MSE.png", dpi=300)


# %%
