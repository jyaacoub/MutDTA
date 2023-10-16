#%%
from src.utils.loader import Loader

dataset = Loader.load_dataset('davis', 'nomsa', 'af2-anm', subset='train')
print(dataset)
# %%
