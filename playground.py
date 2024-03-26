#%%
from src.analysis.figures import prepare_df

df = prepare_df()

# %%
df[df.edge.str.contains('ring3') | df.edge.str.contains('aflow')]

#%%
from src.analysis.figures import fig5_edge_feat_violin

fig5_edge_feat_violin(df, sel_dataset='PDBbind', exclude=['simple'], 
                        sel_col='cindex', show=True, add_stats=False)
fig5_edge_feat_violin(df, sel_dataset='PDBbind', exclude=['simple'], 
                        sel_col='mse', show=True, add_stats=False)

# %%
