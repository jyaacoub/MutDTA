# %%
from src.data_analysis.figures import fig2_pro_feat, fig3_edge_feat, prepare_df

df = prepare_df('results/model_media/model_stats.csv')

# %%
fig2_pro_feat(df, sel_col='pearson')
fig3_edge_feat(df, exclude=['af2-anm'], sel_col='pearson')
# %%
