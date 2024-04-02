# %%
from src.analysis.figures import prepare_df

df = prepare_df()

# %%
import seaborn as sns
def fig7_lig_feat(df, sel_dataset='davis', verbose=False, sel_col='cindex', exclude=[],
                            show=False, add_labels=True, add_stats=True, ax=None):
    filtered_df = df[(df['feat'] == 'nomsa') & (~df['overlap']) & (df['lig_feat'].isna())]
    filtered_df = filtered_df[(filtered_df['data'] == sel_dataset) & (filtered_df['fold'] != '')]

    filtered_df.sort_values(by=['edge'], inplace=True)

    # Dynamically collect values for each edge type not in exclude
    edge_types = filtered_df['edge'].unique()
    edge_types = [e for e in edge_types if e not in exclude]
    plot_data = [filtered_df[filtered_df['edge'] == edge][sel_col] for edge in edge_types]

    # plot violin plot with annotations
    ax = sns.violinplot(data=plot_data, ax=ax)
    ax.set_xticklabels(edge_types)
    ax.set_ylabel(sel_col)
    ax.set_xlabel('Edge type')
    ax.set_title(f'Edge type {sel_col} for {sel_dataset}')

    if add_stats:
        pairs = [(i, j) for i in range(len(edge_types)) for j in range(i+1, len(edge_types))]
        annotator = Annotator(ax, pairs, data=plot_data, verbose=verbose)
        annotator.configure(test='Mann-Whitney', text_format='star', loc='inside', 
                            hide_non_significant=not verbose)
        annotator.apply_and_annotate()
        
    if show:
        plt.show()