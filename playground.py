# %%
import logging
from typing import OrderedDict

import seaborn as sns
from matplotlib import pyplot as plt
from statannotations.Annotator import Annotator

from src.analysis.figures import prepare_df, custom_fig, fig_combined

df = prepare_df()
sel_dataset = 'PDBbind'
exclude = []
sel_col = 'cindex'
# %%

# models to plot:
# - Original model with (nomsa, binary) and (original,  binary) features for protein and ligand respectively
# - Aflow models with   (nomsa, aflow*) and (original,  binary) # x2 models here (aflow and aflow_ring3)
# - GVP protein model   (gvp,   binary) and (original,  binary)
# - GVP ligand model    (nomsa, binary) and (gvp,       binary)

models = {
    'DG': ('nomsa', 'binary', 'original', 'binary'),
    'aflow': ('nomsa', 'aflow', 'original', 'binary'),
    'aflow_ring3': ('nomsa', 'aflow_ring3', 'original', 'binary'),
    # 'gvpP': ('gvp', 'binary', 'original', 'binary'),
    'gvpL': ('nomsa', 'binary', 'gvp', 'binary'),
    'gvpL_aflow': ('nomsa', 'aflow', 'gvp', 'binary'),
    'gvpL_aflow_rng3': ('nomsa', 'aflow_ring3', 'gvp', 'binary'),
}

# custom_fig(df, models, sel_dataset, sel_col)

# %%
fig, axes = fig_combined(df, datasets=['PDBbind'], fig_callable=custom_fig,
             models=models, metrics=['pearson', 'cindex', 'mse', 'mae'],
             fig_scale=(8,5))
plt.xticks(rotation=45)


# %%
