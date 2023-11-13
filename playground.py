# # %%
# from src.data_processing.init_dataset import create_datasets

# create_datasets(
#     data_opt=['davis'],
#     feat_opt=['foldseek'],
#     edge_opt=['binary']
# )
# # %%
# from src.utils.loader import Loader
# d2 = Loader.load_dataset(data='davis', pro_feature='nomsa',
#                         edge_opt='anm')
# %%
from src.data_analysis.figures import fig4_pro_feat_violin, prepare_df

df = prepare_df('results/model_media/model_stats.csv')

#%%
fig4_pro_feat_violin(df, sel_dataset='davis', sel_col='mse',
                     add_stats=True, verbose=False)
# %%
