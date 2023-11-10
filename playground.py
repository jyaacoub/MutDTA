# %%
from src.data_analysis.figures import prepare_df
import seaborn as sns
from statannotations.Annotator import Annotator


df = prepare_df(csv_p='/home/jyaacoub/projects/MutDTA/results/model_media/model_stats.csv')

# %%
# Figure 5: violin plot with error bars for Cross-validation results to show significance among edge feats
sel_dataset='davis'
verbose=False
sel_col='mse'
exclude=[]
show=True
add_labels=True
add_stats=True

# %%
filtered_df = df[(df['feat'] == 'nomsa') & (~df['overlap']) & (df['lig_feat'].isna())]
filtered_df = filtered_df[(filtered_df['data'] == sel_dataset) & (filtered_df['fold'] != '')]

filtered_df.sort_values(by=['edge'], inplace=True)

#%%
binary = filtered_df[filtered_df['edge'] == 'binary'][sel_col]
simple = filtered_df[filtered_df['edge'] == 'simple'][sel_col]
anm = filtered_df[filtered_df['edge'] == 'anm'][sel_col]
af2 = filtered_df[filtered_df['edge'] == 'af2'][sel_col]

# %%
ax = sns.violinplot(data=[binary, simple, anm, af2])
ax.set_xticklabels(['binary', 'simple', 'anm', 'af2'])
ax.set_ylabel(sel_col)
ax.set_xlabel('Edge type')
ax.set_title(f'Edge type {sel_col} for {sel_dataset}')

if add_stats:
    pairs = [('binary', 'simple'), ('binary', 'anm'), ('binary', 'af2'), 
             ('simple', 'anm'), ('simple', 'af2'), ('anm', 'af2')]
    annotator = Annotator(ax, pairs, data=filtered_df, x='edge', y=sel_col, verbose=verbose)
    annotator.configure(test='Mann-Whitney', text_format='star', loc='inside', hide_non_significant=True)
    annotator.apply_and_annotate()
# %%
