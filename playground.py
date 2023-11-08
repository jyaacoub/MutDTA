# %%
from src.data_analysis.figures import prepare_df, fig3_edge_feat

df = prepare_df('results/model_media/model_stats.csv')

# %% fig 2 with pro_feat
fig3_edge_feat(df)

# %%
