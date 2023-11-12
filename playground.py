# %%
from src.data_analysis.figures import prepare_df, fig2_pro_feat, fig5_edge_feat_violin
import seaborn as sns
from statannotations.Annotator import Annotator


df = prepare_df(csv_p='/home/jyaacoub/projects/MutDTA/results/model_media/model_stats.csv')

# %%
fig2_pro_feat(df)

# %%
fig5_edge_feat_violin(df, sel_col='pearson')
# %%
