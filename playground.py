#%%
import matplotlib.pyplot as plt
import seaborn as sns
from src.data_analysis.figures import fig6_protein_appearance


sns.set(style="darkgrid")
fig6_protein_appearance()
plt.savefig('results/figures/fig6_protein_appearance_dataset.png', dpi=300)

# %%
