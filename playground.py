# # %%
# from src.data_analysis.figures import fig2_pro_feat, fig3_edge_feat, prepare_df, fig4_pro_feat_violin, fig5_edge_feat_violin

# df = prepare_df('results/model_media/model_stats.csv')
# data = 'kiba'
# # %%
# fig4_pro_feat_violin(df, sel_col='cindex', sel_dataset=data)
# fig4_pro_feat_violin(df, sel_col='mse', sel_dataset=data)

# #%%
# fig5_edge_feat_violin(df, sel_col='cindex', sel_dataset=data)
# fig5_edge_feat_violin(df, sel_col='mse', sel_dataset=data)

# # %%
# fig2_pro_feat(df, sel_col='pearson')
# fig3_edge_feat(df, exclude=['af2-anm'], sel_col='pearson')
#%%
import json
from src.data_analysis.stratify_protein import check_davis_names, kinbase_to_df
from src.utils import config as cfg

#%%
prot_dict = json.load(open(f'{cfg.DATA_ROOT}/davis/proteins.txt', 'r'))
df = kinbase_to_df()

#%%
prots = check_davis_names(prot_dict, df)
