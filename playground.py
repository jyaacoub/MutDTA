# %%
from matplotlib import pyplot as plt
from src.data_analysis.figures import fig0_dataPro_overlap


#%%
for data in ['kiba', 'davis']:
    fig0_dataPro_overlap(data=data)
    plt.savefig(f'results/figures/fig0_pro_overlap_{data}.png', dpi=300, bbox_inches='tight')
    plt.clf()
# %%
