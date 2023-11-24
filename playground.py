# %%
import matplotlib.pyplot as plt
import seaborn as sns 

from src.utils import config as cfg
from src.data_analysis.figures import (prepare_df, fig1_pro_overlap, 
                                       fig2_pro_feat, fig3_edge_feat, 
                                       fig4_pro_feat_violin, fig5_edge_feat_violin)


# %%
df = prepare_df(csv_p=cfg.MODEL_STATS_CSV, old_csv_p="results/model_media/old_model_stats.csv")

#%% display figures
verbose = False

#%% dataset comparisons
for col in ['cindex', 'pearson']:
    fig1_pro_overlap(df, sel_col=col, verbose=verbose, show=False)
    plt.savefig(f"results/figures/fig1_pro_overlap_{col}.png", dpi=300, bbox_inches='tight')
    plt.clf()
    fig2_pro_feat(df, sel_col=col, verbose=verbose, context='paper', add_labels=False, show=False)
    plt.savefig(f"results/figures/fig2_pro_feat_{col}.png", dpi=300, bbox_inches='tight')
    plt.clf()
    fig3_edge_feat(df, sel_col=col, exclude=['af2-anm'], verbose=verbose, context='paper', add_labels=False, show=False)
    plt.savefig(f"results/figures/fig3_edge_feat_{col}.png", dpi=300, bbox_inches='tight')
    plt.clf()

# %% Davis violin plots
sns.set(style="darkgrid")
for dataset in ['davis', 'kiba', 'PDBbind']:
    for col in ['cindex', 'mse']:
        fig4_pro_feat_violin(df, sel_col=col, sel_dataset=dataset, verbose=verbose, show=False)
        plt.savefig(f"results/figures/fig4_pro_feat_violin_{dataset}_{col}.png", dpi=300, bbox_inches='tight')
        plt.clf()
        fig5_edge_feat_violin(df, sel_col=col, sel_dataset=dataset, exclude=['af2-anm'], verbose=verbose, show=False)
        plt.savefig(f"results/figures/fig5_edge_feat_violin_{dataset}_{col}.png", dpi=300, bbox_inches='tight')
        plt.clf()

# %%
