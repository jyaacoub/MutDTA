#%%
from src.data_processing.datasets import PDBbindDataset
from src.utils import config as cfg
import pandas as pd
import matplotlib.pyplot as plt

# d0 = pd.read_csv(f'{cfg.DATA_ROOT}/DavisKibaDataset/davis/nomsa_anm/full/XY.csv', index_col=0)
d0 = pd.read_csv(f'{cfg.DATA_ROOT}/PDBbindDataset/nomsa_anm/full/XY.csv', index_col=0)

d0['len'] = d0.prot_seq.str.len()

# %%
n, bins, patches = plt.hist(d0['len'], bins=20)
# Set labels and title
plt.xlabel('Protein Sequence length')
plt.ylabel('Frequency')
plt.title('Histogram of Protein Sequence length (davis)')

# Add counts to each bin
for count, x, patch in zip(n, bins, patches):
    plt.text(x + 0.5, count, str(int(count)), ha='center', va='bottom')

cutoff= 1500
print(f"Eliminating codes above {cutoff} length would reduce the dataset by: {len(d0[d0['len'] > cutoff])}")
print(f"\t - Eliminates {len(d0[d0['len'] > cutoff].index.unique())} unique proteins")

# %% -d PDBbind -f nomsa -e anm
from src.utils.loader import Loader
d1 = Loader.load_dataset('PDBbind', 'nomsa', 'anm')

# %%
