# # %%
from src.data_analysis.figures import fig2_pro_feat, fig3_edge_feat, prepare_df, fig4_pro_feat_violin, fig5_edge_feat_violin

df = prepare_df('results/model_media/model_stats.csv')
data = 'davis'
verbose=False

fig2 = fig2_pro_feat(df, sel_col='cindex', verbose=verbose)
fig3 = fig3_edge_feat(df, sel_col='cindex', exclude=['af2-anm'], verbose=verbose)

#%%
pdb_f = fig4_pro_feat_violin(df, sel_col='cindex', sel_dataset='davis', verbose=verbose)
pdb_f = fig4_pro_feat_violin(df, sel_col='mse', sel_dataset='davis', verbose=verbose)

# # %% simplified plot for debugging
# from seaborn import violinplot
# from matplotlib import pyplot as plt
# import pandas as pd
# from statannotations.Annotator import Annotator

# new_df = pd.DataFrame({'binary': kiba_e[0], 'simple': kiba_e[1], 'anm': kiba_e[2], 'af2': kiba_e[3]})
# new_df

# #%%
# ax = violinplot(data=kiba_e)
# ax.set_xticklabels(['binary', 'simple', 'anm', 'a2'])

# pairs = [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]
# annotator = Annotator(ax, pairs, data=kiba_e, verbose=verbose)
# annotator.configure(test='Mann-Whitney', text_format='star', loc='inside', 
#                     hide_non_significant=not verbose)
# annotator.apply_and_annotate()



# # %%
# fig4_pro_feat_violin(df, sel_col='cindex', sel_dataset=data, verbose=verbose)
# fig4_pro_feat_violin(df, sel_col='mse', sel_dataset=data, verbose=verbose)

# #%%
# fig5_edge_feat_violin(df, sel_col='cindex', sel_dataset=data, verbose=verbose)
# fig5_edge_feat_violin(df, sel_col='mse', sel_dataset=data, verbose=verbose)

# %%
fig2_pro_feat(df, sel_col='pearson')
fig3_edge_feat(df, exclude=['af2-anm'], sel_col='pearson')
