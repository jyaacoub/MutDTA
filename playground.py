# %%
from src.analysis.figures import prepare_df, fig5_edge_feat_violin, fig4_pro_feat_violin, fig_combined

# %%
df = prepare_df()

# %%
fig_combined(df, datasets=['PDBbind'], metrics=['cindex', 'mse'],
             fig_callable=fig4_pro_feat_violin, show=True,
             exclude=['shannon'])

# fig_combined(df, datasets=['PDBbind'], metrics=['cindex', 'mse'],
#              fig_callable=fig5_edge_feat_violin, show=True,             
#              exclude=['anm', 'af2_anm'])

# %%
