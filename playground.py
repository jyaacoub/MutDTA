#%%
from src.data_analysis.stratify_protein import check_davis_names, kinbase_to_df

df = kinbase_to_df()
# %%
import json
prot_dict = json.load(open('/home/jyaacoub/projects/data/davis/proteins.txt', 'r'))
# %%
# returns a dictionary of davis protein names (keys) and a truple of the protein name, main family, and subgroup (values)
prots = check_davis_names(prot_dict, df)

# %% plot histogram of main families and their counts
import seaborn as sns
import pandas as pd

main_families = [v[1] for v in prots.values()]
main_families = pd.Series(main_families)
sns.histplot(main_families)


# %%
